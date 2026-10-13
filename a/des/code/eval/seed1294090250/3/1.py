from __future__ import annotations

import random
import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_ROT = {"bottom": 0, "right": 1, "top": 2, "left": 3}
ALL_COLORS = tuple(range(10))

GEN_RETRIES = 400
PLACEMENT_RETRIES = 40
CANVAS_MIN = 24
CANVAS_MAX = 36


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_hollow_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    need = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if int(coords.shape[0]) != need:
        return False
    coord_set = {tuple(map(int, xy)) for xy in coords}
    border = (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )
    return coord_set == border


def _seed_anchors(inner_h: int, inner_w: int) -> dict[str, tuple[int, int]]:
    return {
        "top": (0, inner_w // 2),
        "bottom": (inner_h - 1, inner_w // 2),
        "left": (inner_h // 2, 0),
        "right": (inner_h // 2, inner_w - 1),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        for coords in _components(grid == color):
            if not _is_hollow_rectangle(coords):
                continue
            frames.append({"color": int(color), "bbox": _bbox(coords)})
    assert frames, "expected at least one hollow rectangular frame"
    frame_colors = {frame["color"] for frame in frames}
    assert len(frame_colors) == 1, "expected all frames to share one color"
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _parse_frame_seed(
    grid: np.ndarray, bg: int, frame_color: int, bbox: tuple[int, int, int, int]
) -> dict:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_h, inner_w = map(int, inner.shape)
    assert inner_h >= 3 and inner_w >= 3, "frame interior too small"
    seed_coords = np.argwhere(inner != bg)
    assert seed_coords.shape[0] == 1, "expected one seed cell inside each frame"
    sr, sc = map(int, seed_coords[0])
    seed_color = int(inner[sr, sc])
    assert seed_color != frame_color, "seed color must differ from frame color"
    anchors = _seed_anchors(inner_h, inner_w)
    matches = [side for side, pos in anchors.items() if pos == (sr, sc)]
    assert len(matches) == 1, "seed must sit at the midpoint of one inner wall"
    side = matches[0]
    return {
        "bbox": bbox,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "seed_color": seed_color,
        "side": side,
    }


def _find_stamps(
    grid: np.ndarray, bg: int, frame_boxes: list[tuple[int, int, int, int]]
) -> dict[int, dict]:
    outside = grid != bg
    for r0, c0, r1, c1 in frame_boxes:
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    stamps: dict[int, dict] = {}
    for coords in _components(outside):
        values = [int(grid[int(r), int(c)]) for r, c in coords]
        colors = sorted(set(values))
        assert len(colors) == 2, "each external stamp must use one body color plus one marker color"
        counts = {color: values.count(color) for color in colors}
        marker_colors = [color for color, count in counts.items() if count == 1]
        assert len(marker_colors) == 1, "stamp marker color must be a unique singleton"
        marker_color = int(marker_colors[0])
        body_color = int(next(color for color in colors if color != marker_color))

        r0, c0, r1, c1 = _bbox(coords)
        codes = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in coords:
            rr = int(r) - r0
            cc = int(c) - c0
            codes[rr, cc] = 2 if int(grid[int(r), int(c)]) == marker_color else 1

        marker_cells = np.argwhere(codes == 2)
        assert marker_cells.shape[0] == 1, "stamp marker must occupy exactly one cell"
        mr, mc = map(int, marker_cells[0])
        assert mr == codes.shape[0] - 1, "stamp marker must lie on the bottom edge"
        assert codes.shape[1] % 2 == 1 and mc == codes.shape[1] // 2, "stamp marker must sit at bottom midpoint"
        assert marker_color not in stamps, "marker colors must uniquely identify stamps"
        stamps[marker_color] = {"codes": codes, "body_color": body_color}

    assert stamps, "expected at least one external stamp"
    return stamps


def _stamp_into_frame(
    out: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    side: str,
    codes: np.ndarray,
    body_color: int,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    rotated = np.rot90(codes, SIDE_TO_ROT[side])
    marker_cells = np.argwhere(rotated == 2)
    assert marker_cells.shape[0] == 1
    mr, mc = map(int, marker_cells[0])
    seed_r, seed_c = _seed_anchors(inner_h, inner_w)[side]
    base_r = int(r0 + 1 + seed_r - mr)
    base_c = int(c0 + 1 + seed_c - mc)
    for rr, cc in np.argwhere(rotated > 0):
        gr = base_r + int(rr)
        gc = base_c + int(cc)
        assert r0 < gr < r1 and c0 < gc < c1, "stamp must fit inside the frame interior"
        out[gr, gc] = int(body_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each external stamp to a frame by seed color, rotate it according to the seed wall,
    and stamp it inside the frame using the seed cell as the anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    frames = _find_frames(g, bg)
    frame_color = int(frames[0]["color"])
    frame_infos = [_parse_frame_seed(g, bg, frame_color, frame["bbox"]) for frame in frames]
    stamps = _find_stamps(g, bg, [info["bbox"] for info in frame_infos])

    seed_colors = {info["seed_color"] for info in frame_infos}
    assert seed_colors == set(stamps), "seed colors and stamp marker colors must match exactly"

    out = g.copy()
    for info in frame_infos:
        stamp = stamps[info["seed_color"]]
        _stamp_into_frame(out, info["bbox"], info["side"], stamp["codes"], stamp["body_color"])
    return out


def _template(rows: list[str]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


STAMP_TEMPLATES = (
    _template(["111", "010", "020"]),
    _template(["100", "111", "020"]),
    _template(["001", "111", "020"]),
    _template(["101", "111", "010", "020"]),
    _template(["11111", "00100", "00200"]),
    _template(["10101", "11111", "00100", "00200"]),
    _template(["11011", "01110", "00100", "00200"]),
    _template(["00100", "01110", "11111", "00100", "00200"]),
)


def _odd_at_least(n: int) -> int:
    return int(n if n % 2 == 1 else n + 1)


def _render_frame_patch(
    bg: int, frame_color: int, seed_color: int, inner_h: int, inner_w: int, side: str
) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    sr, sc = _seed_anchors(inner_h, inner_w)[side]
    patch[1 + sr, 1 + sc] = seed_color
    return patch


def _render_stamp_patch(bg: int, template: np.ndarray, body_color: int, marker_color: int) -> np.ndarray:
    patch = np.full(template.shape, bg, dtype=int)
    patch[template == 1] = body_color
    patch[template == 2] = marker_color
    return patch


def _can_place(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    hh, ww = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _occupy(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    hh, ww = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    used[rr0:rr1, cc0:cc1] = True


def _place_rectangles(objects: list[dict], h: int, w: int, pad: int = 1) -> list[tuple[int, int]] | None:
    used = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    order = sorted(range(len(objects)), key=lambda i: objects[i]["patch"].shape[0] * objects[i]["patch"].shape[1], reverse=True)
    for idx in order:
        ph, pw = map(int, objects[idx]["patch"].shape)
        candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(candidates)
        placed = None
        for r0, c0 in candidates:
            if _can_place(used, r0, c0, ph, pw, pad):
                placed = (int(r0), int(c0))
                _occupy(used, r0, c0, ph, pw, pad)
                break
        if placed is None:
            return None
        placements[idx] = placed
    return [pos for pos in placements if pos is not None]


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    mask = patch != bg
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _random_template() -> np.ndarray:
    template = random.choice(STAMP_TEMPLATES)
    if random.random() < 0.5:
        template = np.fliplr(template)
    return template.copy()


def _expected_output(inp: np.ndarray, frame_records: list[dict]) -> np.ndarray:
    out = inp.copy()
    for record in frame_records:
        _stamp_into_frame(
            out,
            record["bbox"],
            record["side"],
            record["template"],
            record["body_color"],
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    bg, frame_color = random.sample(ALL_COLORS, 2)
    n_frames = random.randint(4, 5)
    seed_palette = [color for color in ALL_COLORS if color not in {bg, frame_color}]
    seed_colors = random.sample(seed_palette, n_frames)

    directions = list(SIDES)
    if n_frames > 4:
        directions.append(random.choice(SIDES))
    random.shuffle(directions)

    objects: list[dict] = []
    frame_records: list[dict] = []
    for seed_color, side in zip(seed_colors, directions):
        template = _random_template()
        rotated = np.rot90(template, SIDE_TO_ROT[side])
        rot_h, rot_w = map(int, rotated.shape)
        inner_h = max(5, _odd_at_least(rot_h + random.choice((0, 2))))
        inner_w = max(5, _odd_at_least(rot_w + random.choice((0, 2))))
        body_choices = [color for color in ALL_COLORS if color not in {bg, frame_color, seed_color}]
        body_color = int(random.choice(body_choices))

        frame_patch = _render_frame_patch(bg, frame_color, seed_color, inner_h, inner_w, side)
        stamp_patch = _render_stamp_patch(bg, template, body_color, seed_color)

        frame_idx = len(objects)
        objects.append({"kind": "frame", "patch": frame_patch})
        stamp_idx = len(objects)
        objects.append({"kind": "stamp", "patch": stamp_patch})
        frame_records.append(
            {
                "frame_idx": frame_idx,
                "stamp_idx": stamp_idx,
                "side": side,
                "template": template,
                "body_color": body_color,
            }
        )

    largest_h = max(int(obj["patch"].shape[0]) for obj in objects)
    largest_w = max(int(obj["patch"].shape[1]) for obj in objects)
    footprint = sum((int(obj["patch"].shape[0]) + 2) * (int(obj["patch"].shape[1]) + 2) for obj in objects)
    base_side = max(CANVAS_MIN, largest_h + 2, largest_w + 2, int(np.ceil(np.sqrt(footprint))))

    for _ in range(PLACEMENT_RETRIES):
        h = random.randint(base_side, min(CANVAS_MAX, max(base_side, base_side + 5)))
        min_w = max(base_side, int(np.ceil(footprint / h)))
        if min_w > CANVAS_MAX:
            continue
        w = random.randint(min_w, CANVAS_MAX)
        placements = _place_rectangles(objects, h, w, pad=1)
        if placements is None:
            continue

        inp = np.full((h, w), bg, dtype=int)
        for obj, (r0, c0) in zip(objects, placements):
            _paint_patch(inp, obj["patch"], r0, c0, bg)

        for record in frame_records:
            fr, fc = placements[record["frame_idx"]]
            frame_patch = objects[record["frame_idx"]]["patch"]
            fh, fw = map(int, frame_patch.shape)
            record["bbox"] = (int(fr), int(fc), int(fr + fh - 1), int(fc + fw - 1))

        expected = _expected_output(inp, frame_records)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
