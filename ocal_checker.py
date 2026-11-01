#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocal_checker.py —— 根据 OCAL 屏幕截图分析牛反(牛顿反射镜)光轴问题，给出诊断和修正建议。

原理
----
OCAL 在画面上叠加了纯色参考图形（默认：圈1=绿、圈3=蓝、准星=洋红）。
本程序按颜色把这些图形抠出来，拟合出圆心/半径，再结合真实画面里的
副镜边缘(椭圆度)、主镜中心标记位置，判断问题出在：
  相机/准星基准 → 副镜轴向位置 → 副镜旋转/支架横向偏移 → 副镜 45° 角/支架/调焦座 → 副镜倾斜 → 主镜倾斜

用法
----
  python ocal_checker.py shot.png                       # 自动分析
  python ocal_checker.py shot.png --primary 945 598     # 指定主镜中心标记的像素坐标(截图坐标)
  python ocal_checker.py shot.png --pick                # 弹窗用鼠标点主镜中心标记 / 副镜边缘
  python ocal_checker.py --watch ~/Desktop --notify     # 监视截图目录，新截图自动分析并弹通知(macOS)

依赖: opencv-python, numpy
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

# ---- 判定阈值（单位：占蓝圈半径的百分比 / 椭圆度）。可按自己的望远镜微调 ----
TOL_CAM = 1.5       # 准星中心 vs 蓝圈中心
TOL_AXIAL = 2.0     # 绿圈 vs 蓝圈，沿镜筒轴向（牛反副镜本来就应略偏，所以放宽）
TOL_LATERAL = 1.5   # 绿圈 vs 蓝圈，垂直轴向
ELL_OK = 0.02       # 椭圆度 = 1 - 短轴/长轴
ELL_BAD = 0.05
PRI_OK = 2.0        # 主镜中心标记 vs 准星中心
PRI_BAD = 5.0


class AnalysisError(Exception):
    pass


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise AnalysisError(f"无法读取图片: {path}")
    return img


def color_mask(img_bgr: np.ndarray, rgb, tol: int) -> np.ndarray:
    target = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.int16)
    diff = np.abs(img_bgr.astype(np.int16) - target).max(axis=2)
    return diff < tol


def longest_run(flags: np.ndarray):
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return None
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1)
    run = max(runs, key=len)
    return int(run[0]), int(run[-1])


def find_crosshair(mag: np.ndarray):
    """返回 (cx, cy, (x0, y0, x1, y1))，后者是准星线段覆盖的范围，用来裁出相机画面区域。"""
    rows = mag.sum(axis=1).astype(float)
    cols = mag.sum(axis=0).astype(float)
    k = np.ones(5)
    y0 = int(np.convolve(rows, k, "same").argmax())
    x0 = int(np.convolve(cols, k, "same").argmax())
    if rows[y0] < 100 or cols[x0] < 100:
        return None

    def refine(profile, c):
        lo, hi = max(0, c - 5), min(len(profile), c + 6)
        w = profile[lo:hi]
        return float((np.arange(lo, hi) * w).sum() / max(w.sum(), 1e-9))

    cy, cx = refine(rows, y0), refine(cols, x0)
    rx = longest_run(mag[max(0, y0 - 3): y0 + 4, :].any(axis=0))
    ry = longest_run(mag[:, max(0, x0 - 3): x0 + 4].any(axis=1))
    if rx is None or ry is None:
        return None
    return cx, cy, (rx[0], ry[0], rx[1], ry[1])


def circle_from_3(p):
    (x1, y1), (x2, y2), (x3, y3) = p
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return ux, uy, math.hypot(x1 - ux, y1 - uy)


def kasa_fit(pts: np.ndarray):
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x * x + y * y
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(c[0]), float(c[1]), float(math.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 0)))


def ransac_circle(mask: np.ndarray, iters=400, thr=3.0, rmin=30):
    ys, xs = np.nonzero(mask)
    pts = np.c_[xs, ys].astype(float)
    if len(pts) < 150:
        return None
    rng = np.random.default_rng(1)
    sub = pts if len(pts) <= 5000 else pts[rng.choice(len(pts), 5000, replace=False)]
    best, best_cnt = None, 0
    for _ in range(iters):
        c = circle_from_3(sub[rng.choice(len(sub), 3, replace=False)])
        if c is None or c[2] < rmin:
            continue
        res = np.abs(np.hypot(sub[:, 0] - c[0], sub[:, 1] - c[1]) - c[2])
        cnt = int((res < thr).sum())
        if cnt > best_cnt:
            best, best_cnt = c, cnt
    if best is None:
        return None
    for _ in range(3):
        res = np.abs(np.hypot(pts[:, 0] - best[0], pts[:, 1] - best[1]) - best[2])
        inl = pts[res < thr]
        if len(inl) < 50:
            break
        best = kasa_fit(inl)
    return best


def circ_median(a: np.ndarray, k=31):
    pad = k // 2
    ext = np.r_[a[-pad:], a, a[:pad]]
    return np.array([np.median(ext[i: i + k]) for i in range(len(a))])


# --------------------------------------------------------------------------- #
# 副镜边缘（真实画面里的椭圆）
# --------------------------------------------------------------------------- #
def detect_secondary_ellipse(img, green, overlay_mask):
    """沿绿圈附近做极坐标径向梯度搜索，找副镜真实边缘并拟合椭圆。"""
    cx, cy, r = green
    pad = int(r * 1.4)
    x0, x1 = max(0, int(cx - pad)), min(img.shape[1], int(cx + pad))
    y0, y1 = max(0, int(cy - pad)), min(img.shape[0], int(cy + pad))
    crop = img[y0:y1, x0:x1].copy()
    m = cv2.dilate(overlay_mask[y0:y1, x0:x1].astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    crop = cv2.inpaint(crop, m, 5, cv2.INPAINT_TELEA)  # 抹掉叠加的彩色圈，避免把它们当成边缘
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 2.0)

    n_ang = 360
    rs = np.arange(0.80 * r, 1.25 * r, 0.5)
    th = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    mapx = (cx - x0 + np.outer(np.cos(th), rs)).astype(np.float32)
    mapy = (cy - y0 + np.outer(np.sin(th), rs)).astype(np.float32)
    polar = cv2.remap(gray, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    d = np.abs(np.gradient(polar, axis=1))
    d = cv2.GaussianBlur(d, (0, 0), sigmaX=2, sigmaY=2)

    idx = d.argmax(axis=1)
    rad = rs[idx]
    strength = d.max(axis=1) / (np.median(d) + 1e-6)
    med = circ_median(rad)
    keep = (np.abs(rad - med) < 0.04 * r) & (strength > 3.0)
    if keep.sum() < n_ang * 0.35:
        return None
    px = cx + rad[keep] * np.cos(th[keep])
    py = cy + rad[keep] * np.sin(th[keep])
    pts = np.c_[px, py].astype(np.float32)
    if len(pts) < 5:
        return None
    return ellipse_info(pts, coverage=float(keep.mean()))


def ellipse_info(pts: np.ndarray, coverage=1.0):
    (ex, ey), (w, h), ang = cv2.fitEllipse(pts.astype(np.float32))
    major, minor = max(w, h), min(w, h)
    major_ang = ang if w >= h else ang + 90.0
    return {
        "center": (ex, ey),
        "major": major,
        "minor": minor,
        "ellipticity": 1.0 - minor / major if major > 0 else 0.0,
        "major_angle": major_ang % 180.0,
        "coverage": coverage,
    }


# --------------------------------------------------------------------------- #
# 手动取点（可选）
# --------------------------------------------------------------------------- #
def pick_points(img):
    h, w = img.shape[:2]
    scale = min(1.0, 1300.0 / max(h, w))
    base = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    st = {"primary": None, "skip": False, "edge": []}

    def on_mouse(ev, x, y, flags, param):
        p = (x / scale, y / scale)
        if ev == cv2.EVENT_LBUTTONDOWN:
            if st["primary"] is None and not st["skip"]:
                st["primary"] = p
            else:
                st["edge"].append(p)
        elif ev == cv2.EVENT_RBUTTONDOWN:
            if st["edge"]:
                st["edge"].pop()
            elif st["primary"] is not None:
                st["primary"] = None

    win = "OCAL checker"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("【取点】1) 左键点主镜中心标记(小圆点)；按 s 可跳过这一步")
    print("        2) 再沿副镜真实边缘左键点 6~12 个点；右键撤销")
    print("        3) 回车结束，Esc 放弃")
    while True:
        disp = base.copy()
        if st["primary"] is not None:
            cv2.circle(disp, (int(st["primary"][0] * scale), int(st["primary"][1] * scale)), 9, (0, 165, 255), 2)
        for p in st["edge"]:
            cv2.circle(disp, (int(p[0] * scale), int(p[1] * scale)), 4, (0, 255, 255), -1)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k in (10, 13):
            break
        if k == 27:
            st = {"primary": None, "skip": True, "edge": []}
            break
        if k == ord("s"):
            st["skip"] = True
    cv2.destroyWindow(win)
    return st


# --------------------------------------------------------------------------- #
# 核心分析
# --------------------------------------------------------------------------- #
def vec_words(dx: float, dy: float) -> str:
    """图像坐标(y向下)向量 → 中文方向描述"""
    parts = []
    if abs(dx) >= 0.5:
        parts.append(f"{'右' if dx > 0 else '左'}{abs(dx):.0f}px")
    if abs(dy) >= 0.5:
        parts.append(f"{'下' if dy > 0 else '上'}{abs(dy):.0f}px")
    return "、".join(parts) if parts else "几乎为0"


def dir_name(ux: float, uy: float) -> str:
    if abs(ux) > abs(uy) * 2:
        return "右" if ux > 0 else "左"
    if abs(uy) > abs(ux) * 2:
        return "下" if uy > 0 else "上"
    return ("右" if ux > 0 else "左") + ("下" if uy > 0 else "上")


def analyze(img: np.ndarray, args):
    h, w = img.shape[:2]
    masks = {
        "green": color_mask(img, args.green_rgb, args.tol),
        "blue": color_mask(img, args.blue_rgb, args.tol),
        "cross": color_mask(img, args.cross_rgb, args.tol),
    }
    cross = find_crosshair(masks["cross"])
    if cross is None:
        raise AnalysisError("没找到准星线。请确认截图里有洋红色十字，且颜色与 --cross-rgb 一致。")
    cx, cy, (rx0, ry0, rx1, ry1) = cross

    roi = np.zeros((h, w), bool)  # 只在相机画面区域内找圈，避开右侧控制面板的颜色色块
    roi[ry0: ry1 + 1, rx0: rx1 + 1] = True
    green = ransac_circle(masks["green"] & roi)
    blue = ransac_circle(masks["blue"] & roi)
    if green is None or blue is None:
        raise AnalysisError("没找到绿圈或蓝圈。请在 OCAL 里启用圈1(绿)与圈3(蓝)，并让它们完整出现在截图里。")
    if green[2] > blue[2]:
        green, blue = blue, green  # 兜底：小的当副镜圈

    Rb = blue[2]
    scale = Rb / args.r3  # 截图像素 / OCAL 参数像素

    # --- 轴向分解 ---
    a = math.radians(args.axis_angle)
    ux, uy = math.cos(a), -math.sin(a)  # 镜筒轴向在图像中的单位向量(y向下)
    px_, py_ = -uy, ux                   # 垂直于轴向
    d_gb = np.array([green[0] - blue[0], green[1] - blue[1]])
    d_cb = np.array([cx - blue[0], cy - blue[1]])
    axial = float(d_gb @ np.array([ux, uy]))
    lateral = float(d_gb @ np.array([px_, py_]))

    m = {
        "scale": scale,
        "ratio": green[2] / Rb,
        "expected_ratio": args.r1 / args.r3,
        "cam_vec": d_cb,
        "cam_pct": float(np.hypot(*d_cb) / Rb * 100),
        "gb_vec": d_gb,
        "axial": axial,
        "lateral": lateral,
        "axial_pct": abs(axial) / Rb * 100,
        "lateral_pct": abs(lateral) / Rb * 100,
        "axis_dir": dir_name(ux, uy),
        "ellipse": None,
        "ellipse_src": None,
        "primary": None,
        "primary_vec": None,
        "primary_pct": None,
    }

    # --- 手动 / 自动副镜椭圆与主镜中心 ---
    primary = tuple(args.primary) if args.primary else None
    manual_edge = None
    if args.pick:
        st = pick_points(img)
        if st["primary"] is not None:
            primary = st["primary"]
        if len(st["edge"]) >= 5:
            manual_edge = np.array(st["edge"], np.float32)

    if manual_edge is not None:
        m["ellipse"] = ellipse_info(manual_edge)
        m["ellipse_src"] = "手动取点"
    elif not args.no_ellipse:
        overlay = masks["green"] | masks["blue"] | masks["cross"]
        try:
            e = detect_secondary_ellipse(img, green, overlay)
        except cv2.error:
            e = None
        if e is not None:
            m["ellipse"], m["ellipse_src"] = e, "自动检测"

    if primary is not None:
        v = np.array([primary[0] - cx, primary[1] - cy], float)
        m["primary"] = primary
        m["primary_vec"] = v
        m["primary_pct"] = float(np.hypot(*v) / Rb * 100)

    annotated = annotate(img, green, blue, (cx, cy), m)
    return m, annotated


def annotate(img, green, blue, cross_c, m):
    out = img.copy()
    thin = max(1, img.shape[1] // 1500)
    cv2.circle(out, (int(green[0]), int(green[1])), int(green[2]), (255, 255, 255), thin)
    cv2.circle(out, (int(blue[0]), int(blue[1])), int(blue[2]), (255, 255, 255), thin)
    s = int(blue[2] * 0.04)
    for (x, y), col in (((green[0], green[1]), (0, 255, 255)), ((blue[0], blue[1]), (255, 255, 0))):
        cv2.drawMarker(out, (int(x), int(y)), col, cv2.MARKER_CROSS, s * 2, thin + 1)
    cv2.drawMarker(out, (int(cross_c[0]), int(cross_c[1])), (255, 255, 255), cv2.MARKER_TILTED_CROSS, s * 2, thin + 1)
    e = m["ellipse"]
    if e:
        cv2.ellipse(out, ((e["center"][0], e["center"][1]), (e["major"], e["minor"]), e["major_angle"]),
                    (0, 255, 255), thin + 1)
    if m["primary"] is not None:
        p = (int(m["primary"][0]), int(m["primary"][1]))
        cv2.circle(out, p, max(8, s), (0, 165, 255), thin + 2)
        cv2.arrowedLine(out, p, (int(cross_c[0]), int(cross_c[1])), (0, 165, 255), thin + 1, tipLength=0.15)
    return out


# --------------------------------------------------------------------------- #
# 诊断规则
# --------------------------------------------------------------------------- #
def diagnose(m, args):
    """返回 (issues, actions)。issues: [(level, 标题, 详情)]; level ∈ ok/warn/bad"""
    issues, actions = [], []
    sc = m["scale"]

    # 提示：圈半径比例与参数是否一致，不一致说明 --r1/--r3 没填对，截图换算会不准
    if abs(m["ratio"] - m["expected_ratio"]) > 0.03:
        issues.append(("warn", "圈半径比例与参数不符",
                       f"截图中 绿/蓝 半径比={m['ratio']:.3f}，而 r1/r3={m['expected_ratio']:.3f}。"
                       "请用 --r1 / --r3 传入 OCAL 里的真实数值，否则像素换算会有偏差。"))

    # 1) 相机 / 准星基准
    cam = m["cam_pct"]
    cv_ = m["cam_vec"]
    if cam <= TOL_CAM:
        issues.append(("ok", "准星中心与蓝圈(调焦座)同心", f"偏差 {cam:.1f}% 蓝圈半径。"))
    else:
        issues.append(("warn" if cam < 4 else "bad", "准星中心偏离调焦座轴线",
                       f"准星相对蓝圈圆心偏 {vec_words(cv_[0], cv_[1])}（{cam:.1f}%）。"
                       "这是相机/接口不同心，不是镜子问题——这一步不修正，后面怎么调都对不准。"))
        actions.append(
            f"【基准】勾选 OCAL「启用圆心偏移」，把准星挪到调焦座真实轴线上："
            f"上下约 {abs(round(cv_[1] / sc))}、左右约 {abs(round(cv_[0] / sc))}（OCAL 像素；符号以实际移动方向为准）。"
            "再把相机转 90°/180° 各看一次，十字不漂移才算基准可靠。")

    # 2) 副镜(绿圈) vs 调焦座(蓝圈) 同心
    ax, la = m["axial_pct"], m["lateral_pct"]
    conc_bad = False
    if ax <= TOL_AXIAL and la <= TOL_LATERAL:
        issues.append(("ok", "副镜圈与调焦座圈基本同心",
                       f"轴向 {ax:.1f}%，横向 {la:.1f}%（牛反副镜本就应略偏向主镜，轴向留一点偏差正常）。"))
    else:
        conc_bad = True
        if ax > TOL_AXIAL:
            issues.append(("warn" if ax < 4 else "bad", "副镜轴向位置不对（高度/前后）",
                           f"沿轴向（图中朝{m['axis_dir']}方向）偏 {ax:.1f}%。"))
            actions.append(
                "【副镜位置】松开副镜中心螺丝，左手托住支架防止转动，沿镜筒轴向前后微移副镜（每次约 0.5mm），"
                "观察绿圈圆心朝蓝圈圆心靠拢，直到轴向偏差落在 2% 内。倾斜螺丝先退到行程中间再动。")
        if la > TOL_LATERAL:
            issues.append(("warn" if la < 3 else "bad", "副镜横向偏移或旋转不对",
                           f"垂直轴向偏 {la:.1f}%。"))
            actions.append(
                "【副镜旋转】微调副镜绕轴旋转；若转动后横向偏差始终改不掉，说明支架中心偏离镜筒轴线："
                "量三条臂长度、支架中心到镜筒两侧壁的距离，必要时在偏短一侧垫垫片。")

    # 3) 椭圆度
    e = m["ellipse"]
    ell_bad = False
    if e is None:
        issues.append(("warn", "未能测出副镜真实边缘",
                       "副镜边缘对比度不足或被遮挡。可加 --pick 手动点边缘，或换更清晰的截图。"))
    else:
        el = e["ellipticity"]
        tag = f"（{m['ellipse_src']}，边缘覆盖 {e['coverage'] * 100:.0f}%，长轴方向约 {e['major_angle']:.0f}°）"
        if el <= ELL_OK:
            issues.append(("ok", "副镜近似正圆", f"椭圆度 {el * 100:.1f}% {tag}"))
        else:
            ell_bad = True
            issues.append(("warn" if el < ELL_BAD else "bad", "副镜呈椭圆",
                           f"椭圆度 {el * 100:.1f}%（长轴/短轴={e['major'] / e['minor']:.3f}）{tag}"))
            if conc_bad:
                actions.append("【椭圆】先把上面的位置/旋转偏差修完再判断椭圆——位置偏会把椭圆放大，别提前动倾斜螺丝。")
            else:
                actions.append(
                    "【椭圆】同心已达标仍呈椭圆：先做相机旋转测试——把相机转 90°，椭圆长轴若跟着相机转，"
                    "是广角镜头畸变/相机不正，不用管；若长轴方向不变，才是副镜相对光轴不是 45°。"
                    "后者检查：副镜背面与支架接触面是否贴实、三颗螺丝是否对角交替拧紧、支架安装面角度（量角器/水平仪）。")

    # 4) 主镜中心标记 vs 准星
    pri = m["primary_pct"]
    pri_bad = False
    if pri is None:
        issues.append(("warn", "未提供主镜中心标记位置",
                       "加 --primary X Y（截图坐标）或 --pick 用鼠标点，才能判断主镜/副镜倾斜。"))
    else:
        pv = m["primary_vec"]
        if pri <= PRI_OK:
            issues.append(("ok", "主镜中心标记已对准准星", f"偏差 {pri:.1f}%。"))
        else:
            pri_bad = True
            issues.append(("warn" if pri < PRI_BAD else "bad", "主镜中心标记偏离准星",
                           f"标记在准星 {vec_words(pv[0], pv[1])}（{pri:.1f}%）。"))
            actions.append(
                f"【倾斜】用副镜三颗倾斜螺丝，把主镜中心标记朝准星移动"
                f"（需向{vec_words(-pv[0], -pv[1])}）。每次只动一颗，看标记移动方向，找到各螺丝对应的方向后再组合调整。"
                "副镜这一步只对准「主镜中心」；之后再用主镜背后三颗螺丝把准星对准副镜里的反射影像。")

    # 5) 组合模式：用户遇到的“二选一”
    crit = (1 if conc_bad else 0) + (1 if ell_bad else 0) + (1 if pri_bad else 0)
    if pri_bad and (conc_bad or ell_bad):
        issues.append(("bad", "出现“二选一”特征：同心/椭圆/主镜中心不能同时满足",
                       "三颗倾斜螺丝只能补偿小角度，补偿不了位置偏差。这是支架或调焦座的几何问题，不是调得不够细。"))
        actions.append(
            "【几何排查，按这个顺序】\n"
            "      ① 先把三颗倾斜螺丝都退到行程中间；若调到哪颗已接近到底，说明偏差超出补偿范围。\n"
            "      ② 调焦座垂直度：拆下副镜，量调焦座底座左右两端到镜筒前端口的距离是否相等；水平仪比对接口面与镜筒参考面。\n"
            "      ③ 三臂支架：量三个固定点到镜筒前端口的距离（应一致，误差 <0.5mm）、三条臂长度是否相等；"
            "你改过原厂孔位，要重点看新旧孔位是否有高低差。\n"
            "      ④ 副镜安装面 45° 角与贴合度。\n"
            "      发现某一点偏低，就在对应固定点垫薄垫片，修好后整套重新校准。")
    if crit == 0 and cam <= TOL_CAM and e is not None and pri is not None:
        actions.append("光轴几何看起来良好：可以进入主镜精调，再用星点离焦环（星点测试）做最终确认。")

    # 固定提醒：调整顺序
    actions.append("【顺序提醒】每一轮都按：基准 → 副镜轴向位置 → 副镜旋转 → 副镜倾斜 → 主镜倾斜；"
                   "不要颠倒，也不要用倾斜螺丝去硬补位置偏差。")
    return issues, actions


def render_report(path_label, m, issues, actions):
    icon = {"ok": "✅", "warn": "⚠️ ", "bad": "❌"}
    lines = [f"\n====== OCAL 光轴分析：{path_label} ======"]
    lines.append(f"[测量] 蓝圈半径 {m['ratio'] and ''}换算比例 {m['scale']:.3f} 截图px/OCALpx；"
                 f"绿/蓝半径比 {m['ratio']:.3f}")
    gb = m["gb_vec"]
    lines.append(f"[测量] 绿圈相对蓝圈：{vec_words(gb[0], gb[1])}（轴向 {m['axial_pct']:.1f}% / 横向 {m['lateral_pct']:.1f}%）")
    lines.append("")
    lines.append("— 诊断 —")
    for level, title, detail in issues:
        lines.append(f"{icon[level]} {title}\n     {detail}")
    lines.append("")
    lines.append("— 修正建议（按顺序） —")
    for i, a in enumerate(actions, 1):
        lines.append(f"{i}. {a}")
    return "\n".join(lines)


def summary_for_notify(issues):
    for level in ("bad", "warn"):
        for lv, title, _ in issues:
            if lv == level:
                return title
    return "光轴看起来不错"


def notify_mac(title, message):
    msg = message.replace('"', "'")
    ttl = title.replace('"', "'")
    try:
        subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "{ttl}"'], check=False)
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------- #
# 运行入口
# --------------------------------------------------------------------------- #
def run_file(path: Path, args, notify=False):
    img = imread_unicode(path)
    m, annotated = analyze(img, args)
    issues, actions = diagnose(m, args)
    print(render_report(path.name, m, issues, actions))

    out = Path(args.out) if args.out else path.with_name(path.stem + "_analysis.png")
    ok, buf = cv2.imencode(".png", annotated)
    if ok:
        buf.tofile(str(out))
        print(f"\n标注图已保存: {out}")
    if notify:
        notify_mac("OCAL 光轴检查", summary_for_notify(issues))


def watch(folder: Path, args):
    folder = folder.expanduser()
    print(f"监视目录: {folder}（Ctrl+C 退出）")
    seen = None
    while True:
        files = [p for p in folder.iterdir()
                 if p.suffix.lower() in IMG_EXT and not p.stem.endswith("_analysis")]
        if files:
            newest = max(files, key=lambda p: p.stat().st_mtime)
            key = (newest.name, newest.stat().st_mtime)
            if key != seen:
                seen = key
                time.sleep(0.8)  # 等截图写完
                try:
                    run_file(newest, args, notify=args.notify)
                except AnalysisError as ex:
                    print(f"[跳过 {newest.name}] {ex}")
                    if args.notify:
                        notify_mac("OCAL 光轴检查", str(ex)[:60])
        time.sleep(1.5)


def parse_rgb(s: str):
    v = [int(x) for x in s.split(",")]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("格式: R,G,B")
    return tuple(v)


def main():
    ap = argparse.ArgumentParser(description="根据 OCAL 截图分析牛反光轴并给出修正建议")
    ap.add_argument("image", nargs="?", help="OCAL 屏幕截图")
    ap.add_argument("--watch", metavar="DIR", help="监视目录，新截图自动分析")
    ap.add_argument("--notify", action="store_true", help="分析后弹 macOS 通知")
    ap.add_argument("--primary", nargs=2, type=float, metavar=("X", "Y"), help="主镜中心标记的截图像素坐标")
    ap.add_argument("--pick", action="store_true", help="弹窗用鼠标点主镜中心标记和副镜边缘")
    ap.add_argument("--axis-angle", type=float, default=90.0,
                    help="镜筒轴向在图像中的方向(度，0=向右，90=向上)，默认 90")
    ap.add_argument("--r1", type=float, default=270, help="OCAL 圈1(绿)半径参数")
    ap.add_argument("--r3", type=float, default=349, help="OCAL 圈3(蓝)半径参数")
    ap.add_argument("--green-rgb", type=parse_rgb, default=(0, 255, 0))
    ap.add_argument("--blue-rgb", type=parse_rgb, default=(0, 0, 255))
    ap.add_argument("--cross-rgb", type=parse_rgb, default=(255, 0, 255))
    ap.add_argument("--tol", type=int, default=60, help="颜色匹配容差(0-255)")
    ap.add_argument("--no-ellipse", action="store_true", help="跳过自动椭圆检测")
    ap.add_argument("--out", help="标注图输出路径")
    args = ap.parse_args()

    try:
        if args.watch:
            watch(Path(args.watch), args)
        elif args.image:
            run_file(Path(args.image).expanduser(), args, notify=args.notify)
        else:
            ap.print_help()
            return 1
    except AnalysisError as ex:
        print(f"错误: {ex}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
