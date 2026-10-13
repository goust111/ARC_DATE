from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import Color, find_components
except Exception:  # pragma: no cover - fallback keeps SourceFileLoader robust
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors=None,
        min_area: int = 1,
        return_format: str = "dict",
    ):
        from collections import deque

        if bg is None:
            vals, counts = np.unique(grid, return_counts=True)
            bg = int(vals[np.argmax(counts)])
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [
            (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
        ]
        if colors is None:
            colors = sorted(set(map(int, grid.ravel())) - {int(bg)})
        out = []
        for color in colors:
            color = int(color)
            seen = np.zeros_like(grid, dtype=bool)
            for r, c in np.argwhere(grid == color):
                r = int(r)
                c = int(c)
                if seen[r, c]:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                coords = []
                while q:
                    rr, cc = q.popleft()
                    coords.append((rr, cc))
                    for dr, dc in dirs:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                if len(coords) < int(min_area):
                    continue
                arr = np.array(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append({"color": color, "coords": arr, "bbox": (int(r0), int(c0), int(r1), int(c1)), "area": int(len(coords))})
        return out


FRAME_COLOR = Color.GREY
BG = Color.BLACK
MARKER_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
SIDE_ORDER = ("top", "left", "bottom", "right")
SIDE_RANK = {side: i for i, side in enumerate(SIDE_ORDER)}
MAX_GEN_ATTEMPTS = 400


def _is_frame_component(grid: np.ndarray, bbox: tuple[int, int, int, int], area: int) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    expected = 2 * (h + w) - 4
    if int(area) != expected:
        return False
    patch = grid[r0:r1 + 1, c0:c1 + 1]
    if not np.all(patch[0, :] == FRAME_COLOR):
        return False
    if not np.all(patch[-1, :] == FRAME_COLOR):
        return False
    if not np.all(patch[:, 0] == FRAME_COLOR):
        return False
    if not np.all(patch[:, -1] == FRAME_COLOR):
        return False
    return not np.any(patch[1:-1, 1:-1] == FRAME_COLOR)


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    comps = find_components(
        grid,
        bg=BG,
        colors=[FRAME_COLOR],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        if _is_frame_component(grid, bbox, int(comp["area"])):
            frames.append(bbox)
    frames.sort()
    return frames


def _scan_markers_for_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bbox
    markers: list[dict] = []
    if r0 > 0:
        found = [(r0 - 1, c, int(grid[r0 - 1, c])) for c in range(c0 + 1, c1) if int(grid[r0 - 1, c]) not in (BG, FRAME_COLOR)]
        assert len(found) <= 1
        for r, c, color in found:
            markers.append({"side": "top", "coord": (r, c), "color": color})
    if c0 > 0:
        found = [(r, c0 - 1, int(grid[r, c0 - 1])) for r in range(r0 + 1, r1) if int(grid[r, c0 - 1]) not in (BG, FRAME_COLOR)]
        assert len(found) <= 1
        for r, c, color in found:
            markers.append({"side": "left", "coord": (r, c), "color": color})
    if r1 + 1 < grid.shape[0]:
        found = [(r1 + 1, c, int(grid[r1 + 1, c])) for c in range(c0 + 1, c1) if int(grid[r1 + 1, c]) not in (BG, FRAME_COLOR)]
        assert len(found) <= 1
        for r, c, color in found:
            markers.append({"side": "bottom", "coord": (r, c), "color": color})
    if c1 + 1 < grid.shape[1]:
        found = [(r, c1 + 1, int(grid[r, c1 + 1])) for r in range(r0 + 1, r1) if int(grid[r, c1 + 1]) not in (BG, FRAME_COLOR)]
        assert len(found) <= 1
        for r, c, color in found:
            markers.append({"side": "right", "coord": (r, c), "color": color})
    markers.sort(key=lambda m: (SIDE_RANK[m["side"]], m["coord"][0], m["coord"][1], m["color"]))
    return markers


def _extract_problem(grid: np.ndarray) -> list[dict]:
    frames = _find_frames(grid)
    assert 2 <= len(frames) <= 3
    all_markers = {(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere((grid != BG) & (grid != FRAME_COLOR))}
    used_markers = set()
    problem = []
    for bbox in frames:
        markers = _scan_markers_for_frame(grid, bbox)
        assert 1 <= len(markers) <= 2
        used_markers.update((m["coord"][0], m["coord"][1], m["color"]) for m in markers)
        problem.append({"bbox": bbox, "markers": markers})
    assert used_markers == all_markers
    return problem


def _side_length(bbox: tuple[int, int, int, int], side: str) -> int:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    return inner_w if side in ("top", "bottom") else inner_h


def _priority_key(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    return (_side_length(bbox, side), SIDE_RANK[side])


def _cells_for_marker(bbox: tuple[int, int, int, int], side: str, marker_coord: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mr, mc = marker_coord
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    cells: set[tuple[int, int]] = set()
    if side == "top":
        for c in range(c0 + 1, c1):
            cells.add((r0 + 1, c))
        target_r = r0 + 1 + (inner_h - 1) // 2
        for r in range(r0 + 1, target_r + 1):
            cells.add((r, mc))
    elif side == "bottom":
        for c in range(c0 + 1, c1):
            cells.add((r1 - 1, c))
        target_r = r0 + 1 + inner_h // 2
        for r in range(target_r, r1):
            cells.add((r, mc))
    elif side == "left":
        for r in range(r0 + 1, r1):
            cells.add((r, c0 + 1))
        target_c = c0 + 1 + (inner_w - 1) // 2
        for c in range(c0 + 1, target_c + 1):
            cells.add((mr, c))
    elif side == "right":
        for r in range(r0 + 1, r1):
            cells.add((r, c1 - 1))
        target_c = c0 + 1 + inner_w // 2
        for c in range(target_c, c1):
            cells.add((mr, c))
    else:
        raise ValueError(side)
    return cells


def _paint_frame(out: np.ndarray, bbox: tuple[int, int, int, int], markers: list[dict]) -> None:
    winners: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    for marker in markers:
        key = _priority_key(bbox, marker["side"])
        for cell in _cells_for_marker(bbox, marker["side"], marker["coord"]):
            if cell not in winners or key < winners[cell][0]:
                winners[cell] = (key, int(marker["color"]))
    for (r, c), (_key, color) in winners.items():
        out[r, c] = color


def _render_expected_from_problem(grid: np.ndarray, problem: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for frame in problem:
        for marker in frame["markers"]:
            mr, mc = marker["coord"]
            out[mr, mc] = BG
    for frame in problem:
        _paint_frame(out, frame["bbox"], frame["markers"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each exterior frame marker along the touched inner wall and project it inward to the near center line.

    Input structure:
    - Black background contains 2-3 non-touching grey hollow rectangular frames of thickness 1.
    - Each frame has 1-2 singleton colored markers placed orthogonally adjacent to one side, never on a corner.
    - Marker colors are ordinary ARC colors distinct from the grey frame and black background.

    Rule:
    - Detect which frame side each marker touches, then erase the exterior marker pixel.
    - Inside that frame, fill the whole touched inner wall row or column with the marker color.
    - Also project a 1-pixel line inward from the marker's contact position to the frame's center row or center column; for even inner sizes use the middle line closer to the touching side.
    - When two markers in the same frame overlap, the marker from the shorter contacted side wins; equal side lengths break toward top, then left, then bottom, then right. Frame walls stay unchanged.

    Key insight:
    - A marker encodes both a side choice and a precise contact coordinate, while the frame's geometry fixes how far the projection can extend and which overlaps dominate.
    """
    g = np.asarray(grid, dtype=int)
    problem = _extract_problem(g)
    return _render_expected_from_problem(g, problem)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _marker_coord_from_spec(bbox: tuple[int, int, int, int], marker: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    pos = int(marker["pos"])
    side = marker["side"]
    if side == "top":
        return (r0 - 1, c0 + 1 + pos)
    if side == "bottom":
        return (r1 + 1, c0 + 1 + pos)
    if side == "left":
        return (r0 + 1 + pos, c0 - 1)
    if side == "right":
        return (r0 + 1 + pos, c1 + 1)
    raise ValueError(side)


def _build_problem_from_specs(specs: list[dict]) -> list[dict]:
    problem = []
    for spec in specs:
        markers = []
        for marker in spec["markers"]:
            markers.append(
                {
                    "side": marker["side"],
                    "coord": _marker_coord_from_spec(spec["bbox"], marker),
                    "color": int(marker["color"]),
                }
            )
        markers.sort(key=lambda m: (SIDE_RANK[m["side"]], m["coord"][0], m["coord"][1], m["color"]))
        problem.append({"bbox": spec["bbox"], "markers": markers})
    problem.sort(key=lambda f: f["bbox"])
    return problem


def _random_dim(min_inner: int = 2, max_inner: int = 5) -> int:
    return random.randint(min_inner, max_inner) + 2


def _choose_short_feature() -> dict:
    while True:
        inner_h = random.randint(2, 5)
        inner_w = random.randint(2, 5)
        if inner_h == inner_w:
            continue
        h = inner_h + 2
        w = inner_w + 2
        if inner_h < inner_w:
            pair = random.choice([("top", "right"), ("bottom", "left")])
        else:
            pair = random.choice([("top", "right"), ("bottom", "left")])
        markers = [{"side": pair[0]}, {"side": pair[1]}]
        return {"h": h, "w": w, "markers": markers, "mode": "short"}


def _choose_tie_feature() -> dict:
    inner = random.randint(2, 5)
    pair = random.choice([("top", "right"), ("left", "bottom")])
    return {"h": inner + 2, "w": inner + 2, "markers": [{"side": pair[0]}, {"side": pair[1]}], "mode": "tie"}


def _choose_parallel_feature() -> dict:
    inner = random.choice([3, 5])
    pair = random.choice([("top", "bottom"), ("left", "right")])
    return {"h": inner + 2, "w": inner + 2, "markers": [{"side": pair[0]}, {"side": pair[1]}], "mode": "parallel"}


def _choose_even_frame() -> dict:
    if random.random() < 0.5:
        inner_h = random.choice([2, 4])
        inner_w = random.randint(2, 5)
        side = random.choice(["top", "bottom"])
    else:
        inner_h = random.randint(2, 5)
        inner_w = random.choice([2, 4])
        side = random.choice(["left", "right"])
    return {"h": inner_h + 2, "w": inner_w + 2, "markers": [{"side": side}], "mode": "even"}


def _choose_random_frame() -> dict:
    h = _random_dim()
    w = _random_dim()
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    markers = [{"side": sides[0]}]
    if random.random() < 0.55:
        if random.random() < 0.5:
            pair = random.choice([("top", "right"), ("top", "left"), ("bottom", "right"), ("bottom", "left"), ("top", "bottom"), ("left", "right")])
            markers = [{"side": pair[0]}, {"side": pair[1]}]
        else:
            markers = [{"side": sides[0]}, {"side": sides[1]}]
    return {"h": h, "w": w, "markers": markers, "mode": "random"}


def _assign_positions(spec: dict) -> None:
    inner_h = spec["h"] - 2
    inner_w = spec["w"] - 2
    if spec["mode"] == "parallel":
        if {m["side"] for m in spec["markers"]} == {"top", "bottom"}:
            pos = random.randrange(inner_w)
            for marker in spec["markers"]:
                marker["pos"] = pos
        else:
            pos = random.randrange(inner_h)
            for marker in spec["markers"]:
                marker["pos"] = pos
        return

    for marker in spec["markers"]:
        if marker["side"] in ("top", "bottom"):
            marker["pos"] = random.randrange(inner_w)
        else:
            marker["pos"] = random.randrange(inner_h)


def _place_specs(specs: list[dict], H: int, W: int) -> bool:
    expanded: list[tuple[int, int, int, int]] = []
    for spec in specs:
        h = int(spec["h"])
        w = int(spec["w"])
        placed = False
        for _ in range(300):
            r0 = random.randint(1, H - h - 1)
            c0 = random.randint(1, W - w - 1)
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            reserve = (r0 - 1, c0 - 1, r0 + h, c0 + w)
            if any(_rects_overlap(reserve, other) for other in expanded):
                continue
            spec["bbox"] = bbox
            expanded.append(reserve)
            placed = True
            break
        if not placed:
            return False
    return True


def _render_input_from_specs(H: int, W: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0, c0:c1 + 1] = FRAME_COLOR
        grid[r1, c0:c1 + 1] = FRAME_COLOR
        grid[r0:r1 + 1, c0] = FRAME_COLOR
        grid[r0:r1 + 1, c1] = FRAME_COLOR
        for marker in spec["markers"]:
            mr, mc = _marker_coord_from_spec(spec["bbox"], marker)
            grid[mr, mc] = int(marker["color"])
    return grid


def _expected_winner(spec: dict, sides: tuple[str, str]) -> int:
    lookup = {m["side"]: m for m in spec["markers"]}
    ordered = sorted((lookup[sides[0]], lookup[sides[1]]), key=lambda m: (_side_length(spec["bbox"], m["side"]), SIDE_RANK[m["side"]]))
    return int(ordered[0]["color"])


def _feature_witness(spec: dict) -> tuple[tuple[int, int], int] | None:
    r0, c0, r1, c1 = spec["bbox"]
    sides = tuple(m["side"] for m in spec["markers"])
    side_set = set(sides)
    if spec["mode"] in {"short", "tie"}:
        if side_set == {"top", "right"}:
            return (r0 + 1, c1 - 1), _expected_winner(spec, ("top", "right"))
        if side_set == {"bottom", "left"}:
            return (r1 - 1, c0 + 1), _expected_winner(spec, ("bottom", "left"))
        if side_set == {"top", "left"}:
            return (r0 + 1, c0 + 1), _expected_winner(spec, ("top", "left"))
        if side_set == {"bottom", "right"}:
            return (r1 - 1, c1 - 1), _expected_winner(spec, ("bottom", "right"))
    if spec["mode"] == "parallel":
        inner_h = spec["h"] - 2
        inner_w = spec["w"] - 2
        if side_set == {"top", "bottom"}:
            pos = spec["markers"][0]["pos"]
            return (r0 + 1 + inner_h // 2, c0 + 1 + pos), _expected_winner(spec, ("top", "bottom"))
        if side_set == {"left", "right"}:
            pos = spec["markers"][0]["pos"]
            return (r0 + 1 + pos, c0 + 1 + inner_w // 2), _expected_winner(spec, ("left", "right"))
    return None


def _even_probe(spec: dict) -> tuple[tuple[int, int], tuple[int, int], int]:
    bbox = spec["bbox"]
    marker = spec["markers"][0]
    cells = _cells_for_marker(bbox, marker["side"], _marker_coord_from_spec(bbox, marker))
    r0, c0, r1, c1 = bbox
    if marker["side"] == "top":
        inner_h = spec["h"] - 2
        target = (r0 + 1 + (inner_h - 1) // 2, c0 + 1 + marker["pos"])
        beyond = (target[0] + 1, target[1])
    elif marker["side"] == "bottom":
        inner_h = spec["h"] - 2
        target = (r0 + 1 + inner_h // 2, c0 + 1 + marker["pos"])
        beyond = (target[0] - 1, target[1])
    elif marker["side"] == "left":
        inner_w = spec["w"] - 2
        target = (r0 + 1 + marker["pos"], c0 + 1 + (inner_w - 1) // 2)
        beyond = (target[0], target[1] + 1)
    else:
        inner_w = spec["w"] - 2
        target = (r0 + 1 + marker["pos"], c0 + 1 + inner_w // 2)
        beyond = (target[0], target[1] - 1)
    assert target in cells
    return target, beyond, int(marker["color"])


def _validate_generated(inp: np.ndarray, specs: list[dict]) -> None:
    assert inp.ndim == 2 and max(inp.shape) <= 30
    problem = _build_problem_from_specs(specs)
    parsed = _extract_problem(inp)
    parsed_sig = sorted((frame["bbox"], tuple((m["side"], m["coord"], m["color"]) for m in frame["markers"])) for frame in parsed)
    spec_sig = sorted((frame["bbox"], tuple((m["side"], m["coord"], m["color"]) for m in frame["markers"])) for frame in problem)
    assert parsed_sig == spec_sig

    expected = _render_expected_from_problem(inp, problem)
    actual = main(inp)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, inp)

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        assert np.all(actual[r0, c0:c1 + 1] == FRAME_COLOR)
        assert np.all(actual[r1, c0:c1 + 1] == FRAME_COLOR)
        assert np.all(actual[r0:r1 + 1, c0] == FRAME_COLOR)
        assert np.all(actual[r0:r1 + 1, c1] == FRAME_COLOR)
        for marker in spec["markers"]:
            mr, mc = _marker_coord_from_spec(spec["bbox"], marker)
            assert int(actual[mr, mc]) == BG

        witness = _feature_witness(spec)
        if witness is not None:
            cell, color = witness
            assert int(actual[cell]) == int(color)

        if spec["mode"] == "even":
            target, beyond, color = _even_probe(spec)
            assert int(actual[target]) == color
            assert int(actual[beyond]) == BG

    inner_pixels = np.count_nonzero((actual != inp) & (expected != FRAME_COLOR))
    assert inner_pixels >= 6


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(16, 22)
    W = random.randint(16, 22)
    feature_mode = random.choice(["short", "tie", "parallel"])
    if feature_mode == "short":
        feature = _choose_short_feature()
    elif feature_mode == "tie":
        feature = _choose_tie_feature()
    else:
        feature = _choose_parallel_feature()

    even_frame = _choose_even_frame()
    specs = [feature, even_frame]
    if random.random() < 0.6:
        specs.append(_choose_random_frame())

    total_markers = sum(len(spec["markers"]) for spec in specs)
    if total_markers > len(MARKER_COLORS):
        return None
    palette = random.sample(MARKER_COLORS, total_markers)
    p = 0
    for spec in specs:
        _assign_positions(spec)
        for marker in spec["markers"]:
            marker["color"] = palette[p]
            p += 1

    specs.sort(key=lambda s: s["h"] * s["w"], reverse=True)
    if not _place_specs(specs, H, W):
        return None

    inp = _render_input_from_specs(H, W, specs)
    try:
        _validate_generated(inp, specs)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
