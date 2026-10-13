from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import numpy as np


# ---------------------------------------------------------------------------
# Shared helper bootstrap
# ---------------------------------------------------------------------------


def _load_arc_utils():
    """Load shared arc_utils from this workspace regardless of import context."""
    for parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("arc_utils_for_solver", candidate)
            assert spec is not None and spec.loader is not None, f"failed to load arc_utils spec: {candidate}"
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("arc_utils not found via project layout")


_arc_utils = _load_arc_utils()
get_bg = _arc_utils.get_bg
find_components = _arc_utils.find_components


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def _is_rect_frame_component(grid: np.ndarray, comp: dict) -> bool:
    """Return True if a monochrome component is exactly a 1-cell-thick rectangle frame."""
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if int(comp["area"]) <= 4:
        return False
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    if int(comp["area"]) != 2 * (h + w) - 4:
        return False

    inner = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(inner[0, :] == color):
        return False
    if not np.all(inner[-1, :] == color):
        return False
    if not np.all(inner[:, 0] == color):
        return False
    if not np.all(inner[:, -1] == color):
        return False
    if (inner[1:-1, 1:-1].size > 0 and np.any(inner[1:-1, 1:-1] == color)):
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate framed-panel contents along singleton pointer colors that address another panel.

    Input structure:
    - Background is the dominant color.
    - Several disjoint rectangular framed panels share the same interior size; each panel
      has one unique frame color.
    - Each panel interior contains arbitrary non-background content and one singleton pointer
      pixel whose color equals another panel's frame color.

    Rule:
    - Read each panel's pointer color and interpret it as the address of the destination panel.
    - Clear all panel interiors.
    - Copy each panel's original interior content, excluding only its pointer pixel, into the
      destination panel interior aligned to the panel's upper-left.
    - Keep all frame pixels and background unchanged; all pointer pixels disappear because interiors
      are rewritten after clearing.

    Key insight:
    - Pointer color is both selector and destination identifier, so panel contents are moved in a
      simultaneous global permutation over a pointer cycle graph (possibly with cycles longer than two).
    """

    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frame_components = [comp for comp in components if _is_rect_frame_component(g, comp)]
    assert frame_components, "expected at least one framed panel"

    panels: list[dict] = []
    for comp in frame_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        r_in0, c_in0, r_in1, c_in1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
        if r_in0 > r_in1 or c_in0 > c_in1:
            continue
        frame_color = int(comp["color"])
        panels.append(
            {
                "color": frame_color,
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (r_in0, c_in0, r_in1, c_in1),
            }
        )

    # Keep only one panel per frame color and dedupe any accidental repeated detections.
    dedup: dict[int, dict] = {}
    for p in panels:
        dedup.setdefault(int(p["color"]), p)
    panels = list(dedup.values())

    assert len(panels) >= 2, "expected at least two framed panels"

    panel_inner_sizes = {(p["inner_bbox"][2] - p["inner_bbox"][0] + 1, p["inner_bbox"][3] - p["inner_bbox"][1] + 1) for p in panels}
    assert len(panel_inner_sizes) == 1, "all panel interiors must share the same size"

    panel_index_by_frame_color = {int(p["color"]): idx for idx, p in enumerate(panels)}
    frame_colors = set(panel_index_by_frame_color.keys())

    for idx, panel in enumerate(panels):
        r0, c0, r1, c1 = map(int, panel["inner_bbox"])
        inner = g[r0 : r1 + 1, c0 : c1 + 1]

        pointer_cells: list[tuple[int, int, int]] = []
        for rr, cc in np.argwhere(inner != bg):
            color = int(inner[tuple(map(int, (rr, cc)))])
            if color != int(panel["color"]) and color in frame_colors:
                pointer_cells.append((int(rr), int(cc), color))

        assert len(pointer_cells) == 1, "each panel must contain exactly one singleton pointer pixel"
        pr, pc, pointer_color = pointer_cells[0]

        moved_content = inner.copy()
        moved_content[pr, pc] = bg

        dest_panel = panel_index_by_frame_color.get(int(pointer_color))
        assert dest_panel is not None and int(dest_panel) != idx, "pointer color must reference another panel"

        panel["pointer_color"] = int(pointer_color)
        panel["dest_panel"] = int(dest_panel)
        panel["content"] = moved_content

    assert all("dest_panel" in p for p in panels), "every panel must expose a destination"
    assert len({p["dest_panel"] for p in panels}) == len(panels), "destination must be a permutation"
    assert len({int(p["pointer_color"]) for p in panels}) == len(panels), "pointer colors must be unique"

    out = g.copy()
    for panel in panels:
        r0, c0, r1, c1 = map(int, panel["inner_bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for panel in panels:
        dest = panels[panel["dest_panel"]]
        dr0, dc0, dr1, dc1 = map(int, dest["inner_bbox"])
        out[dr0 : dr1 + 1, dc0 : dc1 + 1] = panel["content"]

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_BG = 0
_DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PANEL_COLOR_POOL = tuple(i for i in range(1, 10))

_PANEL_COUNT_RANGE = (2, 5)
_INNER_H_RANGE = (3, 7)
_INNER_W_RANGE = (3, 7)
_GENERATION_ATTEMPTS = 400
_LAYOUT_ATTEMPTS = 150
_PANEL_CONTENT_COLOR_COUNT = (2, 4)


def _random_derangement(n: int) -> list[int]:
    if n <= 1:
        return [0]
    if n == 2:
        return [1, 0]

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if all(i != perm[i] for i in range(n)):
            return perm


def _random_pointer_permutation(n: int) -> list[int]:
    """Build a nontrivial permutation with explicit coverage of 2-cycle and long-cycle branches."""
    if n == 1:
        return [0]

    # Long-cycle branch appears frequently for n>=3.
    if n >= 3 and random.random() < 0.60:
        order = list(range(n))
        random.shuffle(order)
        perm = [0] * n
        for i, src in enumerate(order):
            perm[src] = order[(i + 1) % n]
        return perm

    return _random_derangement(n)


def _rects_touch(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int], margin: int = 0) -> bool:
    """Return True if two inclusive rectangles overlap or are within `margin` of each other."""
    a0, a1, a2, a3 = rect_a  # r0, c0, r1, c1
    b0, b1, b2, b3 = rect_b
    if a2 + margin < b0:
        return False
    if b2 + margin < a0:
        return False
    if a3 + margin < b1:
        return False
    if b3 + margin < a1:
        return False
    return True


def _place_panel_tops(panel_count: int, outer_h: int, outer_w: int) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    min_h = max(outer_h + 3, 9)
    min_w = max(outer_w + 3, 9)

    for _ in range(_LAYOUT_ATTEMPTS):
        H = random.randint(min_h, 30)
        W = random.randint(min_w, 30)
        candidates = [(r, c) for r in range(1, H - outer_h) for c in range(1, W - outer_w)]
        if len(candidates) < panel_count:
            continue

        random.shuffle(candidates)
        placed: list[tuple[int, int, int, int]] = []
        for r0, c0 in candidates:
            if len(placed) == panel_count:
                break

            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            rect = (r0, c0, r1, c1)

            if any(_rects_touch(rect, other, margin=1) for other in placed):
                continue

            placed.append(rect)
            if len(placed) == panel_count:
                break

        if len(placed) == panel_count:
            # Stable top-left order keeps generation deterministic enough once sampled.
            return H, W, placed

    return None


def _paint_walker_content(h: int, w: int, colors: list[int]) -> np.ndarray:
    patch = np.full((h, w), _BG, dtype=int)
    strokes = random.randint(1, 3)
    for _ in range(strokes):
        color = int(random.choice(colors))
        r, c = random.randrange(h), random.randrange(w)
        steps = random.randint(max(2, h + w - 2), max(3, h * w // 2 + 2))
        for _ in range(steps):
            patch[r, c] = color
            dr, dc = random.choice(_DIRS)
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                r, c = nr, nc
    return patch


def _paint_lines_content(h: int, w: int, colors: list[int]) -> np.ndarray:
    patch = np.full((h, w), _BG, dtype=int)
    for _ in range(random.randint(1, 3)):
        color = int(random.choice(colors))
        if random.random() < 0.5:
            row = random.randint(0, h - 1)
            c0 = random.randint(0, w - 1)
            c1 = random.randint(c0, w - 1)
            patch[row, c0:c1 + 1] = color
        else:
            col = random.randint(0, w - 1)
            r0 = random.randint(0, h - 1)
            r1 = random.randint(r0, h - 1)
            patch[r0:r1 + 1, col] = color
    return patch


def _paint_dot_content(h: int, w: int, colors: list[int]) -> np.ndarray:
    patch = np.full((h, w), _BG, dtype=int)
    for _ in range(random.randint(2, min(8, h * w // 2))):
        r, c = random.randrange(h), random.randrange(w)
        patch[r, c] = int(random.choice(colors))
    return patch


def _random_content(h: int, w: int, palette: list[int]) -> np.ndarray:
    painter = random.choice((_paint_walker_content, _paint_lines_content, _paint_dot_content))
    content = painter(h, w, palette)
    if not np.any(content != _BG):
        r, c = random.randrange(h), random.randrange(w)
        content[r, c] = int(random.choice(palette))
    return content


def _draw_frame(canvas: np.ndarray, top_left: tuple[int, int], h: int, w: int, frame_color: int, inner: np.ndarray) -> None:
    r0, c0 = map(int, top_left)
    r1 = r0 + h + 1
    c1 = c0 + w + 1
    canvas[r0, c0 : c1 + 1] = frame_color
    canvas[r1, c0 : c1 + 1] = frame_color
    canvas[r0 : r1 + 1, c0] = frame_color
    canvas[r0 : r1 + 1, c1] = frame_color
    canvas[r0 + 1 : r1, c0 + 1 : c1] = inner


def _build_case() -> np.ndarray:
    panel_count = random.randint(*_PANEL_COUNT_RANGE)
    inner_h = random.randint(*_INNER_H_RANGE)
    inner_w = random.randint(*_INNER_W_RANGE)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    # Ensure content colors leave room for all frame colors and still be nontrivial.
    frame_colors = random.sample(list(_PANEL_COLOR_POOL), panel_count)
    layout = _place_panel_tops(panel_count, outer_h, outer_w)
    if layout is None:
        raise RuntimeError("failed to place panels")

    H, W, rects = layout
    rects = rects[:panel_count]

    permutation = _random_pointer_permutation(panel_count)
    pointer_colors = [int(frame_colors[target]) for target in permutation]

    available_colors = [c for c in _PANEL_COLOR_POOL if c not in frame_colors]
    canvas = np.full((H, W), _BG, dtype=int)

    panels: list[dict] = []
    used_pointer_cells: set[tuple[int, int]] = set()
    for idx, (r0, c0, r1, c1) in enumerate(rects):
        palette = random.sample(available_colors, k=random.randint(*_PANEL_CONTENT_COLOR_COUNT))
        if not palette:
            palette = [c for c in _PANEL_COLOR_POOL if c not in frame_colors]

        inner = _random_content(inner_h, inner_w, palette)

        # Pointer may sit on a pattern pixel or background.
        pr, pc = random.randrange(inner_h), random.randrange(inner_w)
        while (r0 + 1 + pr, c0 + 1 + pc) in used_pointer_cells:
            pr, pc = random.randrange(inner_h), random.randrange(inner_w)
        inner[pr, pc] = pointer_colors[idx]
        used_pointer_cells.add((r0 + 1 + pr, c0 + 1 + pc))

        _draw_frame(canvas, (r0, c0), inner_h, inner_w, int(frame_colors[idx]), inner)
        panels.append(
            {
                "idx": idx,
                "frame_color": int(frame_colors[idx]),
                "top_left": (r0, c0),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
        )

    # Ensure main executes on this sample.
    _ = main(canvas)

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_ATTEMPTS):
        try:
            candidate = _build_case()
            out = main(candidate)
            assert out.ndim == 2 and max(out.shape) <= 30
            assert max(candidate.shape) <= 30
            assert not np.array_equal(out, candidate), "expected actual movement under pointer permutation"
            return candidate
        except Exception:
            continue

    raise RuntimeError("failed to generate valid input for fable/3/014/3")
