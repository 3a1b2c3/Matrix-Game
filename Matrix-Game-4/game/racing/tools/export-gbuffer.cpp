// Export exact (not estimated) normal and depth data for an already-recorded take, by replaying its exact
// recorded camera trajectory (captures/<take>/frames.jsonl) through Renderer::renderGeometryBuffers(). This is
// a new, separate tool: it does not touch MainSDL.cpp, ReferenceServerMain.cpp or RendererGL.cpp's existing
// render()/readBack() paths, so the live game and the normal offline RGB recording are both unaffected.
//
// Correctness note: static geometry (buildings, roads, parked dressing) gets exact normals/depth, since it
// only depends on the recorded camera. The player's own car is replayed too: takes start partway into an
// already-running session (session.json's start_tick), so the car's true starting pose is not simulated from
// World's own spawn -- instead frame 0's recorded camera is algebraically inverted (World::playerCamera's
// fixed eye/target offsets from the car's position+heading, keyed by session.json's camera_mode) to recover
// the car's real starting ground position and heading, placed directly via World::placePlayer(). From there,
// World::update() replays the take's real recorded per-frame input (captures/<take>/inputs.jsonl's "applied"
// events, one per frame) instead of autodriving; physics is deterministic given identical input and starting
// state, so this reproduces the original run's car position/pose frame for frame, with two residual
// approximations: World::playerCamera's cameraCollision() adjustment (if it moved frame 0's recorded eye,
// this cannot invert that) and World::placePlayer()'s resetCar() starting the car at zero velocity rather
// than whatever velocity it actually had at the take's start_tick (physics.resetCar has no velocity
// parameter), so the first fraction of a second may show a brief speed ease-in versus the original. Other
// traffic and pedestrians still do NOT match the original recording: inputs.jsonl only covers the player, and
// no simulation seed was saved, so World::update() drives them along their own independent AI, not a replay.
//
// Writes raw frames only (normal.raw: rgb24, depth.raw: float32), plus depth_meta.json with the exact shape;
// it does not mux a video container itself (no libavformat dependency here). Use ffmpeg/numpy afterward, e.g.
// via scripts/dgx/export_gbuffer_post.sh, to turn normal.raw into normal.mp4 and depth.raw into depth.npy.
//
// Usage:   export-gbuffer --assets resources/racing/assets --frames captures/take-001/frames.jsonl
//                          --output captures/take-001/gbuffer [--inputs captures/take-001/inputs.jsonl]
//                          [--session captures/take-001/session.json] [--fps 48]
// --inputs/--session default to inputs.jsonl/session.json next to --frames.
#include "../src/World.h"
#include "../src/Renderer.h"
#include <nlohmann/json.hpp>
#include <SDL.h>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

using Json=nlohmann::json;

namespace {
std::string next(int& i,int argc,char** argv){if(++i>=argc)throw std::runtime_error("Missing argument value");return argv[i];}

// Inverts World::playerCamera's fixed eye/target offsets (World.cpp) to recover the car's ground
// position and heading from a recorded camera. Returns {ground, heading}; ground already has the
// 0.9f ride-height World::placePlayer() itself adds subtracted back out.
std::pair<V3,float> playerPoseFromCamera(V3 eye,V3 target,int cameraMode){
    V3 forwardXZ=target-eye;forwardXZ.y=0;
    float length=std::sqrt(forwardXZ.x*forwardXZ.x+forwardXZ.z*forwardXZ.z);
    if(length<1e-4f)throw std::runtime_error("camera eye/target too close to recover heading");
    V3 forward=forwardXZ*(1.f/length);
    float heading=std::atan2(forward.x,forward.z);
    V3 p;
    if(cameraMode==0)p=eye+forward*6.9f-V3{0,2.75f,0};
    else if(cameraMode==1)p=eye+forward*6.f-V3{0,2.1f,0};
    else p=eye-forward*1.75f-V3{0,.92f,0};
    return {p-V3{0,.9f,0},heading};
}
}

int main(int argc,char** argv){
try{
    std::filesystem::path assets,framesPath,inputsPath,sessionPath,output;
    double fps=48;
    for(int i=1;i<argc;++i){
        std::string arg=argv[i];
        if(arg=="--assets")assets=next(i,argc,argv);
        else if(arg=="--frames")framesPath=next(i,argc,argv);
        else if(arg=="--inputs")inputsPath=next(i,argc,argv);
        else if(arg=="--session")sessionPath=next(i,argc,argv);
        else if(arg=="--output")output=next(i,argc,argv);
        else if(arg=="--fps")fps=std::stod(next(i,argc,argv));
        else throw std::runtime_error("Unknown argument: "+arg);
    }
    if(assets.empty()||framesPath.empty()||output.empty())
        throw std::runtime_error("Usage: export-gbuffer --assets DIR --frames captures/<take>/frames.jsonl --output DIR [--inputs FILE] [--session FILE] [--fps N]");
    if(inputsPath.empty())inputsPath=framesPath.parent_path()/"inputs.jsonl";
    if(sessionPath.empty())sessionPath=framesPath.parent_path()/"session.json";
    if(std::filesystem::exists(output))throw std::runtime_error("Output directory already exists: "+output.string());
    std::filesystem::create_directories(output);

    std::ifstream sessionFile(sessionPath);
    if(!sessionFile)throw std::runtime_error("Cannot open session file: "+sessionPath.string());
    Json session=Json::parse(sessionFile);
    int trafficCount=session.value("traffic",64),pedestrianCount=session.value("pedestrians",160);
    int cameraMode=session.value("camera_mode",0);

    std::ifstream file(framesPath);
    if(!file)throw std::runtime_error("Cannot open frames file: "+framesPath.string());
    std::vector<Json> frames;
    for(std::string line;std::getline(file,line);)if(!line.empty())frames.push_back(Json::parse(line));
    if(frames.empty())throw std::runtime_error("No frames in "+framesPath.string());
    const int width=864,height=480;

    std::ifstream inputsFile(inputsPath);
    if(!inputsFile)throw std::runtime_error("Cannot open inputs file: "+inputsPath.string());
    std::vector<Input> inputByFrame(frames.size());
    std::vector<bool> inputSet(frames.size(),false);
    for(std::string line;std::getline(inputsFile,line);){
        if(line.empty())continue;
        Json row=Json::parse(line);
        if(row.value("event","")!="applied")continue;
        size_t frame=row.at("next_frame").get<size_t>();
        if(frame>=frames.size())continue;
        const auto& in=row.at("input");
        inputByFrame[frame]={in.at("throttle").get<float>(),in.at("steer").get<float>(),
                              in.at("brake").get<float>(),in.at("handbrake").get<float>()};
        inputSet[frame]=true;
    }
    for(size_t i=0;i<frames.size();++i)
        if(!inputSet[i])throw std::runtime_error("No recorded input for frame "+std::to_string(i)+" in "+inputsPath.string());

    const bool headless=std::getenv("CITYSAMPLE_EGL")&&std::string(std::getenv("CITYSAMPLE_EGL"))=="1";
    if(SDL_Init((headless?0:SDL_INIT_VIDEO)|SDL_INIT_TIMER))throw std::runtime_error(SDL_GetError());
    std::unique_ptr<SDL_Window,decltype(&SDL_DestroyWindow)> window(headless?nullptr:
        SDL_CreateWindow("export-gbuffer",0,0,width,height,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN),SDL_DestroyWindow);
    if(!headless&&!window)throw std::runtime_error(SDL_GetError());
    Renderer renderer(window.get(),width,height);
    World world(assets/"bbox-assets.bin",trafficCount,pedestrianCount);
    world.upload(renderer);

    {
        const auto& cam0=frames[0].at("camera");
        V3 eye0={cam0.at("eye")[0].get<float>(),cam0.at("eye")[1].get<float>(),cam0.at("eye")[2].get<float>()};
        V3 target0={cam0.at("target")[0].get<float>(),cam0.at("target")[1].get<float>(),cam0.at("target")[2].get<float>()};
        auto [ground,heading]=playerPoseFromCamera(eye0,target0,cameraMode);
        world.placePlayer(0,ground,heading);
    }

    std::ofstream normalFile((output/"normal.raw").string(),std::ios::binary);
    if(!normalFile)throw std::runtime_error("Cannot write normal.raw");
    std::ofstream depthFile((output/"depth.raw").string(),std::ios::binary);
    if(!depthFile)throw std::runtime_error("Cannot write depth.raw");

    const float dt=1.f/float(fps);
    for(size_t i=0;i<frames.size();++i){
        // Frame 0's pose came from placePlayer() above (it already reflects frame 0's own recorded
        // input, baked into the physics state at the tick its camera was captured); frames 1..N-1
        // advance one recorded-input step each, same as the original recording's physics ticks.
        if(i>0)world.update(dt,inputByFrame[i],false);
        const auto& cam=frames[i].at("camera");
        Camera camera;
        camera.eye={cam.at("eye")[0].get<float>(),cam.at("eye")[1].get<float>(),cam.at("eye")[2].get<float>()};
        camera.target={cam.at("target")[0].get<float>(),cam.at("target")[1].get<float>(),cam.at("target")[2].get<float>()};
        camera.horizontalFov=cam.at("horizontal_fov").get<float>();

        const auto buffer=renderer.renderGeometryBuffers(world.drawObjects(),camera);
        std::vector<uint8_t> normalRgb(size_t(width)*height*3);
        std::vector<float> depth(size_t(width)*height);
        for(size_t p=0;p<size_t(width)*height;++p){
            normalRgb[p*3+0]=uint8_t(std::clamp(buffer[p*4+0]*.5f+.5f,0.f,1.f)*255.f);
            normalRgb[p*3+1]=uint8_t(std::clamp(buffer[p*4+1]*.5f+.5f,0.f,1.f)*255.f);
            normalRgb[p*3+2]=uint8_t(std::clamp(buffer[p*4+2]*.5f+.5f,0.f,1.f)*255.f);
            depth[p]=std::max(0.f,buffer[p*4+3]);
        }
        normalFile.write(reinterpret_cast<const char*>(normalRgb.data()),std::streamsize(normalRgb.size()));
        depthFile.write(reinterpret_cast<const char*>(depth.data()),std::streamsize(depth.size()*sizeof(float)));
        if(i%50==0)std::cout<<"frame "<<i<<"/"<<frames.size()<<std::endl;
    }
    normalFile.close();depthFile.close();
    Json meta={{"frames",frames.size()},{"width",width},{"height",height},{"fps",fps},
               {"note","Static geometry and the player's own car are exact: the car's starting pose is "
                       "recovered from frame 0's recorded camera and then driven with the take's recorded "
                       "inputs.jsonl, not autodrive. Other traffic/pedestrians are not replayed from the "
                       "original take and will not match it (see file header comment)."}};
    std::ofstream(output/"depth_meta.json")<<meta.dump(2);
    std::cout<<"Done: "<<frames.size()<<" frames -> "<<output.string()<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<"export-gbuffer: "<<e.what()<<std::endl;return 1;}
}
