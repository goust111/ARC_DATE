from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
QUAD_POS = ((0, 0), (0, 1), (1, 0), (1, 1))
MAX_GENERATION_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_frame_component(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    occ = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        occ[r - r0, c - c0] = True
    if not occ[0, :].all() or not occ[-1, :].all():
        return False
    if not occ[:, 0].all() or not occ[:, -1].all():
        return False
    return not occ[1:-1, 1:-1].any()


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            if _is_frame_component(comp):
                frames.append(_bbox(comp))
    assert frames, "expected at least one rectangular frame"
    heights = {r1 - r0 + 1 for r0, _c0, r1, _c1 in frames}
    widths = {c1 - c0 + 1 for _r0, c0, _r1, c1 in frames}
    assert len(heights) == 1 and len(widths) == 1, "all frames must share a size"
    return sorted(frames)


def _quadrant_slices(bbox: tuple[int, int, int, int]) -> list[tuple[slice, slice]]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h > 0 and inner_w > 0 and inner_h % 2 == 0 and inner_w % 2 == 0
    qh = inner_h // 2
    qw = inner_w // 2
    return [
        (slice(r0 + 1, r0 + 1 + qh), slice(c0 + 1, c0 + 1 + qw)),
        (slice(r0 + 1, r0 + 1 + qh), slice(c0 + 1 + qw, c0 + 1 + 2 * qw)),
        (slice(r0 + 1 + qh, r0 + 1 + 2 * qh), slice(c0 + 1, c0 + 1 + qw)),
        (slice(r0 + 1 + qh, r0 + 1 + 2 * qh), slice(c0 + 1 + qw, c0 + 1 + 2 * qw)),
    ]


def _read_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg: int,
) -> tuple[int, list[int | None]]:
    r0, _c0, _r1, c1 = bbox
    assert c1 + 1 < grid.shape[1], "tag pixel must sit immediately right of the frame"
    tag = int(grid[r0, c1 + 1])
    assert tag != bg, "tag pixel must be non-background"

    quads: list[int | None] = []
    for rs, cs in _quadrant_slices(bbox):
        block = grid[rs, cs]
        vals = {int(v) for v in np.unique(block)}
        if vals == {bg}:
            quads.append(None)
            continue
        assert bg not in vals and len(vals) == 1, "each visible quadrant must be a solid color block"
        quads.append(next(iter(vals)))
    return tag, quads


def _fill_pattern(out: np.ndarray, bbox: tuple[int, int, int, int], pattern: np.ndarray) -> None:
    for idx, (rs, cs) in enumerate(_quadrant_slices(bbox)):
        rr, cc = QUAD_POS[idx]
        out[rs, cs] = int(pattern[rr, cc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete tagged four-quadrant frames by copying and rotating the matching color layout.

    Input structure:
    - The background is black.
    - Several same-sized hollow rectangular frames are drawn with a single-pixel grey border.
    - Some frames have all four interior quadrants filled with solid color blocks, while others
      show only one colored quadrant and leave the other three quadrants black.
    - A single non-background tag pixel sits immediately to the right of each frame's top-right corner.

    Rule:
    - Treat every fully filled frame as a labeled 2x2 color pattern, keyed by its outside tag pixel.
    - For each partially filled frame, find the fully filled frame with the same tag color.
    - Rotate that 2x2 pattern by a multiple of 90 degrees so that the partial frame's existing
      colored quadrant matches both the anchor color and the anchor position, then fill the other
      three quadrants with the rotated pattern.

    Key insight:
    - The tag identifies which reference pattern to use, but the visible anchor quadrant is needed
      to infer how that pattern must be rotated before copying it into the incomplete frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)

    reference_by_tag: dict[int, np.ndarray] = {}
    partials: list[tuple[tuple[int, int, int, int], int, int, int]] = []

    for bbox in frames:
        tag, quads = _read_frame(g, bbox, bg)
        visible = [idx for idx, color in enumerate(quads) if color is not None]
        if len(visible) == 4:
            colors = [int(color) for color in quads if color is not None]
            assert len(set(colors)) == 4, "reference frame quadrants must use four distinct colors"
            pattern = np.array([[colors[0], colors[1]], [colors[2], colors[3]]], dtype=int)
            if tag in reference_by_tag:
                assert np.array_equal(reference_by_tag[tag], pattern), "same tag must map to one pattern"
            else:
                reference_by_tag[tag] = pattern
        else:
            assert len(visible) == 1, "partial frame must expose exactly one anchor quadrant"
            anchor_idx = visible[0]
            anchor_color = int(quads[anchor_idx])
            partials.append((bbox, tag, anchor_idx, anchor_color))

    assert reference_by_tag, "expected at least one fully filled reference frame"
    assert partials, "expected at least one partially filled frame"

    out = g.copy()
    for bbox, tag, anchor_idx, anchor_color in partials:
        assert tag in reference_by_tag, "partial frame tag must match a reference frame"
        pattern = reference_by_tag[tag]
        matches: list[np.ndarray] = []
        for k in range(4):
            rotated = np.rot90(pattern, k)
            rr, cc = QUAD_POS[anchor_idx]
            if int(rotated[rr, cc]) == anchor_color:
                matches.append(rotated)
        assert len(matches) == 1, "anchor color and position must determine a unique rotation"
        _fill_pattern(out, bbox, matches[0])
    return out


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_pattern(rng: np.random.Generator, tag: int) -> np.ndarray:
    choices = [color for color in PALETTE.tolist() if color != tag]
    colors = rng.choice(choices, size=4, replace=False)
    return rng.permutation(colors).reshape(2, 2).astype(int)


def _build_layout(
    rng: np.random.Generator,
    frame_h: int,
    frame_w: int,
    n_frames: int,
) -> tuple[int, int, list[tuple[int, int]]]:
    rows = 2
    cols = 2 if n_frames == 4 else 3
    footprint_w = frame_w + 1
    gap_r = int(rng.integers(1, 4))
    gap_c = int(rng.integers(1, 4))
    base_h = rows * frame_h + (rows - 1) * gap_r
    base_w = cols * footprint_w + (cols - 1) * gap_c
    if base_h + 2 > 30 or base_w + 2 > 30:
        raise ValueError("layout exceeds ARC size limit")

    extra_h = int(rng.integers(2, 30 - base_h + 1))
    extra_w = int(rng.integers(2, 30 - base_w + 1))
    top = int(rng.integers(1, extra_h))
    left = int(rng.integers(1, extra_w))
    bottom = extra_h - top
    right = extra_w - left
    assert top >= 1 and bottom >= 1 and left >= 1 and right >= 1

    height = base_h + extra_h
    width = base_w + extra_w
    slots = [
        (top + rr * (frame_h + gap_r), left + cc * (footprint_w + gap_c))
        for rr in range(rows)
        for cc in range(cols)
    ]
    rng.shuffle(slots)
    return height, width, slots[:n_frames]


def _paint_frame_contents(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    pattern: np.ndarray,
    anchor_only: int | None,
) -> None:
    for idx, (rs, cs) in enumerate(_quadrant_slices(bbox)):
        if anchor_only is not None and idx != anchor_only:
            continue
        rr, cc = QUAD_POS[idx]
        grid[rs, cs] = int(pattern[rr, cc])


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    qh = int(rng.integers(1, 4))
    qw = int(rng.integers(1, 4))
    frame_h = 2 + 2 * qh
    frame_w = 2 + 2 * qw

    n_frames = 4
    if frame_w <= 6 and rng.random() < 0.5:
        n_frames = 5

    try:
        height, width, placements = _build_layout(rng, frame_h, frame_w, n_frames)
    except ValueError:
        return None

    tags = rng.choice(PALETTE, size=2, replace=False).astype(int)
    patterns = {int(tag): _sample_pattern(rng, int(tag)) for tag in tags.tolist()}

    descriptors: list[dict[str, int | np.ndarray]] = []
    partial_rots = rng.permutation(4)
    for idx, tag in enumerate(tags.tolist()):
        descriptors.append({"tag": int(tag), "kind": 0, "pattern": patterns[int(tag)]})
        descriptors.append(
            {
                "tag": int(tag),
                "kind": 1,
                "pattern": patterns[int(tag)],
                "rot": int(partial_rots[idx]),
                "anchor": int(rng.integers(0, 4)),
            }
        )
    if n_frames == 5:
        extra_tag = int(tags[int(rng.integers(0, len(tags)))])
        descriptors.append(
            {
                "tag": extra_tag,
                "kind": 1,
                "pattern": patterns[extra_tag],
                "rot": int(rng.integers(0, 4)),
                "anchor": int(rng.integers(0, 4)),
            }
        )
    rng.shuffle(descriptors)

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    for (r0, c0), desc in zip(placements, descriptors):
        bbox = (int(r0), int(c0), int(r0 + frame_h - 1), int(c0 + frame_w - 1))
        tag = int(desc["tag"])
        pattern = np.asarray(desc["pattern"], dtype=int)
        _draw_border(inp, bbox, FRAME_COLOR)
        _draw_border(out, bbox, FRAME_COLOR)
        inp[r0, c0 + frame_w] = tag
        out[r0, c0 + frame_w] = tag

        if int(desc["kind"]) == 0:
            _paint_frame_contents(inp, bbox, pattern, anchor_only=None)
            _paint_frame_contents(out, bbox, pattern, anchor_only=None)
        else:
            rotated = np.rot90(pattern, int(desc["rot"]))
            anchor_idx = int(desc["anchor"])
            _paint_frame_contents(inp, bbox, rotated, anchor_only=anchor_idx)
            _paint_frame_contents(out, bbox, rotated, anchor_only=None)

    if _most_common_color(inp) != BG:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
