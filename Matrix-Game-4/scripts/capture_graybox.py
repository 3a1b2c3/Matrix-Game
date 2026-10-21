"""Record a clip of the EdenGolem taxi game in Matrix-Game's gray-box input style, as an input video for `matrix-game render`.

The look follows Matrix-Game's own recorded takes (captures/take-001/reference.mkv): flat unlit colors per kind of object, thin black
outlines, gray road and light gray sidewalks with white lane dashes, large flat building slabs and no sky or shading, at the default
864x480 render size. The game is stepped on Playwright's fake clock, exactly 1/fps seconds per frame, and driven through real keyboard
input by the held-key rules of a strategy file, so a rerun gives an identical clip. The HUD and result card are hidden.

This script only styles and records; it reuses the capture helpers of the game project (capture.py) and changes nothing in it.
Run it with a Python that has playwright and imageio-ffmpeg and a Playwright Chromium (the EdenLive venv on this PC has all three).

Output: <out>/<id>.mp4 (default captures/<id>.mp4 in this repository). Copy it to the DGX and render it with
scripts/dgx/run_racing_yellow_taxi.sh or scripts/dgx/run_taxi_quality.sh.
"""

import argparse
import functools
import os
import shutil
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GAME = Path(os.environ.get('EDENGOLEM_GAME', r'C:\workspace\world\EdenGolem_games\city_racer')).resolve()
if not (GAME / 'capture.py').is_file():
    raise SystemExit(f'EdenGolem game project not found: {GAME} (set EDENGOLEM_GAME to its folder)')
sys.path.insert(0, str(GAME))

import capture as game_capture  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

DEFAULT_WIDTH, DEFAULT_HEIGHT = 864, 480

# Flat colors per object class, outlines, no lights, no fog, no sky. Applied to the live scene after it loads.
STYLE_JS = """
async () => {
  const THREE = await import('/vendor/three.module.js');
  const scene = window.__scene.scene;
  scene.fog = null;
  scene.background = new THREE.Color(0xa9aba7);
  const palette = { buildings: [0x8b6b9b, 0x8da55a, 0xb87a8a, 0x7f8f6a], backdrop: [0x8b6b9b, 0x8da55a] };
  const fixed = { player: 0x4f7fa0, traffic_car: 0x4a9aa8, roadblock: 0xa87c4f, checkpoint: 0x4fb0b8,
                  finish_line: 0xdddddd, street_props: 0xa8a84a };
  const noOutline = new Set(['road', 'checkpoint', 'finish_line']);
  const topOf = object => { while (object.parent && object.parent !== scene) object = object.parent; return object; };
  const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x101010 });
  const meshes = [];
  scene.traverse(object => { if (object.isMesh) meshes.push(object); });
  const slabColor = new Map();
  let slabs = 0;
  for (const mesh of meshes) {
    const group = topOf(mesh).name;
    const old = mesh.material.color;
    let color;
    if (group === 'road') {
      const luminance = old ? (old.r + old.g + old.b) / 3 : 1;
      color = mesh.isInstancedMesh || mesh.material.isMeshBasicMaterial ? 0xeeeeee : (luminance < 0.28 ? 0x8c8e8c : 0xb0b2ae);
    } else if (palette[group]) {
      let root = mesh;
      while (root.parent && root.parent.name !== group) root = root.parent;
      if (!slabColor.has(root.uuid)) slabColor.set(root.uuid, palette[group][slabs++ % palette[group].length]);
      color = slabColor.get(root.uuid);
    } else {
      color = fixed[group] ?? (old ? old.getHex() : 0xffffff);
    }
    mesh.material = new THREE.MeshBasicMaterial({ color });
    if (!noOutline.has(group) && !mesh.isInstancedMesh) {
      mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 30), edgeMaterial));
    }
  }
  scene.traverse(object => { if (object.isLight) object.visible = false; });
  return { restyled: meshes.length };
}
"""


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Record the taxi game in Matrix-Game gray-box style.')
    parser.add_argument('--id', default='taxi-graybox', help='clip name, used for the output file')
    parser.add_argument('--out', type=Path, default=REPO / 'captures', help='output folder')
    parser.add_argument('--frames', type=int, default=217, help='number of frames (217 = 9 s at 24 fps)')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH)
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT)
    parser.add_argument('--strategy', type=Path, default=GAME / 'capture-strategies' / 'lane-change.json',
                        help='strategy file whose held-key rules drive the taxi')
    parser.add_argument('--scenario', type=int, default=0, help='scenario in the strategy file')
    parser.add_argument('--keep-frames', action='store_true', help='keep the PNG frames next to the video')
    return parser.parse_args(argv)


def record(args, frames_dir):
    rules = game_capture.load_rules(args.strategy, args.scenario)
    port = game_capture.free_port()
    handler = functools.partial(game_capture.Step1Handler, runs_dir=GAME / 'runs/step3', runs_rel='runs/step3',
                                project_root=GAME)
    server = game_capture.ReusableServer(('127.0.0.1', port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    launch = {'headless': True, 'args': game_capture.BROWSER_ARGS}
    chromium = game_capture.find_chromium()
    if chromium:
        launch['executable_path'] = chromium
    errors = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(**launch)
            page = browser.new_page(viewport={'width': args.width, 'height': args.height}, device_scale_factor=1)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.clock.install(time=0)
            page.clock.pause_at(1000)  # from here only run_for advances game time
            page.goto(f'http://127.0.0.1:{port}/runs/step3/acceptance/', wait_until='load')
            page.add_style_tag(content=game_capture.HIDE_OVERLAYS)
            game_capture.wait_for_game(page)
            page.evaluate(STYLE_JS)
            final = game_capture.capture_frames(page, rules, args.frames, args.fps, frames_dir)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    if errors:
        raise SystemExit('page errors during capture: ' + '; '.join(errors))
    return final


def main(argv=None):
    args = parse_args(argv)
    if args.frames < 2 or args.fps < 1:
        raise SystemExit('frames must be at least 2 and fps at least 1')
    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / f'{args.id}_frames'
    shutil.rmtree(frames_dir, ignore_errors=True)
    final = record(args, frames_dir)

    video = args.out / f'{args.id}.mp4'
    game_capture.encode(frames_dir, video, args.fps)
    if not args.keep_frames:
        shutil.rmtree(frames_dir)
    print(f'video: {video}')
    print(f'{args.frames} frames at {args.fps} fps, {args.width}x{args.height}; game state at the end: {final["state"]}, '
          f'z={final["car"]["z"]:.1f}, health={final["health"]}, checkpoints={final["checkpoints"]}, hits={final["hits"]}, '
          f'forward speed={final["car"]["forwardSpeed"]:.1f} m/s')


if __name__ == '__main__':
    main()
