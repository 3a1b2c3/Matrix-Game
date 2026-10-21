// Export exact (not estimated) normal and depth data for an already-recorded take, by replaying its exact
// recorded camera trajectory (captures/<take>/frames.jsonl) through Renderer::renderGeometryBuffers(). This is
// a new, separate tool: it does not touch MainSDL.cpp, ReferenceServerMain.cpp or RendererGL.cpp's existing
// render()/readBack() paths, so the live game and the normal offline RGB recording are both unaffected.
//
// Correctness note: static geometry (buildings, roads, parked dressing) gets exact normals/depth, since it
// only depends on the recorded camera. The player's own car is exact too: every frame places it at the take's
// recorded position/heading (frames.jsonl "position"/"heading", the same pose the recorded camera was derived
// from) via World::placePlayer(); it is not simulated, because the take ran several physics ticks per frame
// and a one-step-per-frame replay drifts away from the recorded camera.
// Other traffic and pedestrians still do NOT match the original recording: no simulation seed was saved, so
// World::update() drives them along their own independent AI, not a replay.
//
// Writes raw frames only (normal.raw: rgb24, depth.raw: float32), plus depth_meta.json with the exact shape;
// it does not mux a video container itself (no libavformat dependency here). Use ffmpeg/numpy afterward, e.g.
// via scripts/dgx/export_gbuffer_post.sh, to turn normal.raw into normal.mp4 and depth.raw into depth.npy.
//
// Usage:   export-gbuffer --assets resources/racing/assets --frames captures/take-001/frames.jsonl
//                          --output captures/take-001/gbuffer
//                          [--session captures/take-001/session.json] [--fps 48]
// --session defaults to session.json next to --frames.
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
}

int main(int argc,char** argv){
try{
    std::filesystem::path assets,framesPath,sessionPath,output;
    double fps=48;
    for(int i=1;i<argc;++i){
        std::string arg=argv[i];
        if(arg=="--assets")assets=next(i,argc,argv);
        else if(arg=="--frames")framesPath=next(i,argc,argv);
        else if(arg=="--session")sessionPath=next(i,argc,argv);
        else if(arg=="--output")output=next(i,argc,argv);
        else if(arg=="--fps")fps=std::stod(next(i,argc,argv));
        else throw std::runtime_error("Unknown argument: "+arg);
    }
    if(assets.empty()||framesPath.empty()||output.empty())
        throw std::runtime_error("Usage: export-gbuffer --assets DIR --frames captures/<take>/frames.jsonl --output DIR [--session FILE] [--fps N]");
    if(sessionPath.empty())sessionPath=framesPath.parent_path()/"session.json";
    if(std::filesystem::exists(output))throw std::runtime_error("Output directory already exists: "+output.string());
    std::filesystem::create_directories(output);

    std::ifstream sessionFile(sessionPath);
    if(!sessionFile)throw std::runtime_error("Cannot open session file: "+sessionPath.string());
    Json session=Json::parse(sessionFile);
    int trafficCount=session.value("traffic",64),pedestrianCount=session.value("pedestrians",160);

    std::ifstream file(framesPath);
    if(!file)throw std::runtime_error("Cannot open frames file: "+framesPath.string());
    std::vector<Json> frames;
    for(std::string line;std::getline(file,line);)if(!line.empty())frames.push_back(Json::parse(line));
    if(frames.empty())throw std::runtime_error("No frames in "+framesPath.string());
    const int width=864,height=480;

    const bool headless=std::getenv("CITYSAMPLE_EGL")&&std::string(std::getenv("CITYSAMPLE_EGL"))=="1";
    if(SDL_Init((headless?0:SDL_INIT_VIDEO)|SDL_INIT_TIMER))throw std::runtime_error(SDL_GetError());
    std::unique_ptr<SDL_Window,decltype(&SDL_DestroyWindow)> window(headless?nullptr:
        SDL_CreateWindow("export-gbuffer",0,0,width,height,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN),SDL_DestroyWindow);
    if(!headless&&!window)throw std::runtime_error(SDL_GetError());
    Renderer renderer(window.get(),width,height);
    World world(assets/"bbox-assets.bin",trafficCount,pedestrianCount);
    world.upload(renderer);

    std::ofstream normalFile((output/"normal.raw").string(),std::ios::binary);
    if(!normalFile)throw std::runtime_error("Cannot write normal.raw");
    std::ofstream depthFile((output/"depth.raw").string(),std::ios::binary);
    if(!depthFile)throw std::runtime_error("Cannot write depth.raw");

    const float dt=1.f/float(fps);
    for(size_t i=0;i<frames.size();++i){
        // The player's car is placed from the take's recorded per-frame position/heading (frames.jsonl), not
        // simulated: the take ran several physics ticks per frame, so a one-step-per-frame replay drifts away
        // from the recorded camera. Traffic/pedestrians are still advanced by World::update().
        world.update(dt,Input{},false);
        const auto& pos=frames[i].at("position");
        world.placePlayer(0,V3{pos[0].get<float>(),pos[1].get<float>()-.9f,pos[2].get<float>()},frames[i].at("heading").get<float>());
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
               {"note","Static geometry and the player's own car are exact: the car is placed every frame at the take's "
                       "recorded position/heading (frames.jsonl). Other traffic/pedestrians are not replayed from "
                       "the original take and will not match it (see file header comment)."}};
    std::ofstream(output/"depth_meta.json")<<meta.dump(2);
    std::cout<<"Done: "<<frames.size()<<" frames -> "<<output.string()<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<"export-gbuffer: "<<e.what()<<std::endl;return 1;}
}
