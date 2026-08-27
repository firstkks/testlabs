"""창 없이 장면을 그림으로 남기는 도우미. 모든 실습 스크립트가 시각화 이미지를 하나 이상 남길 때 쓴다.

- snapshots(): 지정한 시각들에서 프레임을 찍어 가로로 이어 붙인 PNG를 저장한다.
- 카메라 이름을 주지 않으면 MuJoCo의 자유 카메라(모델 중심을 바라봄)를 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import os
import numpy as np
from PIL import Image, ImageDraw

import mujoco


def _ensure_framebuffer(model: mujoco.MjModel, width: int, height: int) -> None:
    """모델의 오프스크린 버퍼(기본 640x480)가 요청 크기보다 작으면 키운다."""
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)


def _label_font(size: int):
    """matplotlib에 딸린 DejaVu Sans를 쓴다(어느 OS에나 있다). 없으면 PIL 기본 글꼴."""
    try:
        import matplotlib
        from PIL import ImageFont
        path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
        return ImageFont.truetype(path, size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    size = max(14, im.width // 30)                       # 900 px 프레임에서 30 px: 전자책 폭에서도 읽힌다
    font = _label_font(size)
    box = d.textbbox((0, 0), text, font=font)
    pad = size // 3
    d.rectangle([8, 8, 8 + (box[2] - box[0]) + 2 * pad, 8 + (box[3] - box[1]) + 2 * pad], fill=(255, 255, 255))
    d.text((8 + pad, 8 + pad - box[1]), text, fill=(20, 20, 20), font=font)
    return np.asarray(im)


def snapshots(model: mujoco.MjModel, data: mujoco.MjData, times, out_path, *, camera: str | None = None,
              step_fn=None, width: int = 900, height: int = 600, labels: bool = True) -> Path:
    """data.time이 times의 각 값에 이를 때까지 step_fn(기본 mj_step)을 돌리며 프레임을 찍는다.
    times의 첫 값이 현재 시각이면 계산만 하고(mj_forward) 찍는다."""
    _ensure_framebuffer(model, width, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    step = step_fn or (lambda: mujoco.mj_step(model, data))
    frames = []
    mujoco.mj_forward(model, data)
    for t in times:
        while data.time < t - 1e-9:
            step()
        if camera:
            renderer.update_scene(data, camera=camera)
        else:
            renderer.update_scene(data)
        f = renderer.render().copy()
        frames.append(_label(f, f"t = {data.time:.2f} s") if labels else f)
    renderer.close()
    strip = np.concatenate(frames, axis=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(out_path)
    return out_path


def poses(model: mujoco.MjModel, data: mujoco.MjData, qpos_list, out_path, *, camera: str | None = None,
          labels=None, width: int = 700, height: int = 600) -> Path:
    """여러 qpos를 차례로 넣고(mj_forward만) 각 자세를 찍어 이어 붙인다. 시간은 흐르지 않는다."""
    _ensure_framebuffer(model, width, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = []
    for i, q in enumerate(qpos_list):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        if camera:
            renderer.update_scene(data, camera=camera)
        else:
            renderer.update_scene(data)
        f = renderer.render().copy()
        frames.append(_label(f, labels[i]) if labels else f)
    renderer.close()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(frames, axis=1)).save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# 실시간 viewer 재생: 모든 실습 스크립트가 --view 옵션으로 같은 장면을 창에서 보여줄 때 쓴다.
# macOS에서는 mjpython으로 실행해야 한다:  uv run mjpython <script>.py --view
# ---------------------------------------------------------------------------
import sys
import time


def track_camera(model: mujoco.MjModel, body: str = "torso", distance: float = 1.4, azimuth: float = 135.0,
                 elevation: float = -18.0) -> mujoco.MjvCamera:
    """몸통을 따라다니는 카메라. 걷는 로봇이 화면을 벗어나지 않게 한다 (snapshots·poses의 camera 인자로 넘긴다)."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body(body).id
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def want_view() -> bool:
    """명령행에 --view가 있으면 True."""
    return "--view" in sys.argv


def view_speed(default: float = 1.0) -> float:
    """--speed 0.25 처럼 재생 배속을 읽는다 (1.0 = 실시간, 0.25 = 1/4 슬로모션)."""
    if "--speed" in sys.argv:
        i = sys.argv.index("--speed")
        if i + 1 < len(sys.argv):
            try:
                return float(sys.argv[i + 1])
            except ValueError:
                pass
    return default


def play(model: mujoco.MjModel, data: mujoco.MjData, *, step_fn=None, duration: float | None = None,
         realtime: bool = True, speed: float | None = None, title: str = "", fps: float = 60.0) -> None:
    """viewer 창을 띄우고 step_fn(기본 mj_step)을 벽시계에 맞춰 돌린다.

    - 화면은 fps(기본 60)로 갱신하고, 물리는 각 프레임에서 "벽시계 × speed"에 시뮬레이션 시각이 따라잡을 만큼 돌린다.
      그래서 스텝마다 sync하던 방식과 달리 실제 시간과 정확히 맞는다.
    - speed: 1.0이 실시간, 0.25면 4배 느린 슬로모션. None이면 명령행 --speed 값(기본 1.0).
    - 창을 닫거나 duration(시뮬레이션 시간)이 지나면 끝나며, 마지막에 실제 배속을 출력한다.
    """
    try:
        import mujoco.viewer
    except Exception as e:  # pragma: no cover
        print("viewer를 열 수 없습니다:", e)
        return
    step = step_fn or (lambda: mujoco.mj_step(model, data))
    if speed is None:
        speed = view_speed()
    if title:
        print(f"[viewer] {title}  (창을 닫으면 종료, 배속 x{speed:g})")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            t0 = data.time
            wall0 = time.perf_counter()
            frame = 1.0 / fps
            while viewer.is_running():
                frame_start = time.perf_counter()
                if realtime:
                    target = t0 + (frame_start - wall0) * speed
                    n = 0
                    while data.time < target and n < 20000:
                        step(); n += 1
                else:
                    step()
                viewer.sync()
                if duration is not None and data.time - t0 >= duration:
                    break
                rest = frame - (time.perf_counter() - frame_start)
                if rest > 0:
                    time.sleep(rest)
            wall = time.perf_counter() - wall0
            if wall > 0:
                print(f"[viewer] 시뮬레이션 {data.time - t0:.2f} s 를 실제 {wall:.2f} s 동안 재생 (배속 x{(data.time - t0) / wall:.2f})")
    except RuntimeError as e:
        msg = str(e)
        if "mjpython" in msg or "main thread" in msg:
            print("macOS에서는 viewer를 띄우려면 mjpython으로 실행하세요:  uv run mjpython <script>.py --view")
        else:
            raise
