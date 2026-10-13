from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

SELECTOR_SIZE = 7
SPRITE_SIZE = 3
MARKER_POSITIONS = ((1, 1), (1, 5), (5, 1), (5, 5))
PAYLOAD_COLORS = (BLUE, GREEN, YELLOW, PINK)
KINDS = ("hline", "vline", "point")
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _find_selector_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for coords in _components_from_mask(grid == GREY):
        box = _bbox(coords)
        if coords.shape[0] == 45 and _bbox_h(box) == SELECTOR_SIZE and _bbox_w(box) == SELECTOR_SIZE:
            candidates.append(box)
    assert len(candidates) == 1, "expected exactly one 7x7 grey selector"
    return candidates[0]


def _classify_local_kind(occupied: np.ndarray, red_rc: tuple[int, int]) -> str | None:
    rr, cc = red_rc
    has_lr = 0 <= cc - 1 and cc + 1 < occupied.shape[1] and occupied[rr, cc - 1] and occupied[rr, cc + 1]
    has_ud = 0 <= rr - 1 and rr + 1 < occupied.shape[0] and occupied[rr - 1, cc] and occupied[rr + 1, cc]
    if has_lr and has_ud:
        return None
    if has_lr:
        return "hline"
    if has_ud:
        return "vline"
    return "point"


def _transform_deltas(deltas: list[tuple[int, int]], kind: str, mode: str) -> list[tuple[int, int]]:
    if mode == "copy":
        return list(deltas)
    if mode == "force_point":
        return [(-dr, -dc) for dr, dc in deltas]
    if mode == "alt_axis":
        if kind == "hline":
            return [(-dr, dc) for dr, dc in deltas]
        if kind == "vline":
            return [(dr, -dc) for dr, dc in deltas]
        return [(-dr, -dc) for dr, dc in deltas]
    if kind == "hline":
        return [(dr, -dc) for dr, dc in deltas]
    if kind == "vline":
        return [(-dr, dc) for dr, dc in deltas]
    return [(-dr, -dc) for dr, dc in deltas]


def _extract_sprite_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    colors = sorted({int(grid[r, c]) for r, c in coords})
    assert RED in colors and len(colors) == 2, "sprite must contain red plus one payload color"
    payload_color = next(color for color in colors if color != RED)
    box = _bbox(coords)
    assert _bbox_h(box) <= SPRITE_SIZE and _bbox_w(box) <= SPRITE_SIZE
    local = np.zeros((_bbox_h(box), _bbox_w(box)), dtype=int)
    red_local: tuple[int, int] | None = None
    r0, c0, _r1, _c1 = box
    payload_deltas: list[tuple[int, int]] = []
    for r, c in coords:
        rr = int(r) - r0
        cc = int(c) - c0
        local[rr, cc] = int(grid[r, c])
        if int(grid[r, c]) == RED:
            red_local = (rr, cc)
    assert red_local is not None
    occupied = local != BG
    kind = _classify_local_kind(occupied, red_local)
    assert kind is not None, "ambiguous plus-shaped marker geometry"
    for r, c in coords:
        if int(grid[r, c]) == RED:
            continue
        payload_deltas.append((int(r) - (r0 + red_local[0]), int(c) - (c0 + red_local[1])))
    return {
        "coords": coords,
        "payload_color": int(payload_color),
        "kind": kind,
        "payload_deltas": payload_deltas,
    }


def _render_output(
    sprites_by_color: dict[int, dict],
    selector_colors: dict[tuple[int, int], int],
    *,
    mode: str = "correct",
) -> np.ndarray:
    out = np.full((SELECTOR_SIZE, SELECTOR_SIZE), GREY, dtype=int)
    for center in MARKER_POSITIONS:
        color = int(selector_colors[center])
        sprite = sprites_by_color[color]
        cr, cc = center
        out[cr, cc] = RED
        for dr, dc in _transform_deltas(sprite["payload_deltas"], sprite["kind"], mode):
            rr = cr + dr
            cc2 = cc + dc
            assert abs(dr) <= 1 and abs(dc) <= 1, "generated sprite does not fit its 3x3 target quadrant"
            assert 0 <= rr < SELECTOR_SIZE and 0 <= cc2 < SELECTOR_SIZE
            out[rr, cc2] = color
    return out


def _solve(grid: np.ndarray, *, mode: str = "correct") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    selector_box = _find_selector_bbox(g)
    sr0, sc0, sr1, sc1 = selector_box

    selector = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    selector_colors = {pos: int(selector[pos]) for pos in MARKER_POSITIONS}
    assert set(selector_colors.values()) == set(PAYLOAD_COLORS), "selector markers must be the four payload colors"
    for r in range(SELECTOR_SIZE):
        for c in range(SELECTOR_SIZE):
            if (r, c) in selector_colors:
                continue
            assert int(selector[r, c]) == GREY

    outside = g.copy()
    outside[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg
    components = _components_from_mask(outside != bg)
    assert len(components) == 4, "expected exactly four external sprites"

    sprites_by_color: dict[int, dict] = {}
    kinds_seen: set[str] = set()
    for coords in components:
        assert 3 <= coords.shape[0] <= 6
        info = _extract_sprite_info(g, coords)
        color = info["payload_color"]
        assert color in PAYLOAD_COLORS and color not in sprites_by_color
        sprites_by_color[color] = info
        kinds_seen.add(info["kind"])

    assert len(sprites_by_color) == 4
    assert len(kinds_seen) >= 2
    return _render_output(sprites_by_color, selector_colors, mode=mode)


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble mirrored color-matched sprites into a 7x7 selector summary.

    Input structure:
    A black background contains one 7x7 grey selector block whose four 3x3 quadrant centers are marked with the
    four payload colors {1,3,4,6}. Elsewhere there are exactly four small 4-connected mixed-color sprites; each
    sprite contains exactly one red pixel and otherwise uses exactly one payload color matching one selector marker.

    Rule:
    Match each sprite to the selector quadrant whose center has the same non-red color. If the red pixel is the
    center of a horizontal 3-pixel line within the sprite, mirror the non-red pixels across the vertical axis through
    the red pixel. If it is the center of a vertical 3-pixel line, mirror the non-red pixels across the horizontal
    axis through the red pixel. Otherwise point-reflect the non-red pixels by 180 degrees around the red pixel. Build
    a 7x7 grey output canvas and place each transformed sprite into its matched 3x3 quadrant with the red pixel
    aligned to that quadrant's center.

    Key insight:
    The selector only tells where each sprite goes; the red geometry embedded inside each sprite independently tells
    how that sprite must be transformed before placement.
    """
    return _solve(grid, mode="correct")


def _is_connected_with_center(cells: set[tuple[int, int]]) -> bool:
    q = deque([(1, 1)])
    seen = {(1, 1)}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen == cells


def _build_template_pools() -> dict[str, list[dict]]:
    around = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
    pools: dict[str, list[dict]] = {kind: [] for kind in KINDS}
    seen: dict[str, set[tuple[tuple[int, int], ...]]] = {kind: set() for kind in KINDS}
    for mask_bits in range(1 << len(around)):
        payload = [around[i] for i in range(len(around)) if (mask_bits >> i) & 1]
        if not (2 <= len(payload) <= 5):
            continue
        occupied = {(1, 1), *payload}
        if not _is_connected_with_center(occupied):
            continue
        occ = np.zeros((3, 3), dtype=bool)
        for r, c in occupied:
            occ[r, c] = True
        kind = _classify_local_kind(occ, (1, 1))
        if kind is None:
            continue
        deltas = sorted((r - 1, c - 1) for r, c in payload)
        transformed = tuple(sorted(_transform_deltas(deltas, kind, "correct")))
        alt = tuple(sorted(_transform_deltas(deltas, kind, "alt_axis")))
        if transformed == tuple(deltas):
            continue
        if kind != "point" and transformed == alt:
            continue
        if transformed in seen[kind]:
            continue
        seen[kind].add(transformed)
        pools[kind].append(
            {
                "kind": kind,
                "payload_deltas": list(deltas),
                "occupied": occupied,
                "transformed": transformed,
            }
        )
    for kind in KINDS:
        assert pools[kind], f"no templates for {kind}"
    return pools


TEMPLATE_POOLS = _build_template_pools()


def _sprite_from_template(template: dict, color: int) -> np.ndarray:
    sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    sprite[1, 1] = RED
    for dr, dc in template["payload_deltas"]:
        sprite[1 + dr, 1 + dc] = int(color)
    return sprite


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_selector_box(h: int, w: int) -> tuple[int, int, int, int]:
    center_r = (h - SELECTOR_SIZE) // 2
    center_c = (w - SELECTOR_SIZE) // 2
    rows = range(max(1, center_r - 2), min(h - SELECTOR_SIZE, center_r + 2) + 1)
    cols = range(max(1, center_c - 2), min(w - SELECTOR_SIZE, center_c + 2) + 1)
    r0 = random.choice(tuple(rows))
    c0 = random.choice(tuple(cols))
    return int(r0), int(c0), int(r0 + SELECTOR_SIZE - 1), int(c0 + SELECTOR_SIZE - 1)


def _place_sprite_boxes(
    h: int,
    w: int,
    selector_box: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int]] | None:
    candidates: list[tuple[int, int]] = []
    for r in range(h - SPRITE_SIZE + 1):
        for c in range(w - SPRITE_SIZE + 1):
            box = (r, c, r + SPRITE_SIZE - 1, c + SPRITE_SIZE - 1)
            if _boxes_separated(box, selector_box, gap=1):
                candidates.append((r, c))
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    chosen_boxes: list[tuple[int, int, int, int]] = []
    for r, c in candidates:
        box = (r, c, r + SPRITE_SIZE - 1, c + SPRITE_SIZE - 1)
        if all(_boxes_separated(box, prev, gap=1) for prev in chosen_boxes):
            chosen.append((r, c))
            chosen_boxes.append(box)
            if len(chosen) == count:
                return chosen
    return None


def _build_selector(selector_colors: dict[tuple[int, int], int]) -> np.ndarray:
    selector = np.full((SELECTOR_SIZE, SELECTOR_SIZE), GREY, dtype=int)
    for pos, color in selector_colors.items():
        selector[pos] = int(color)
    return selector


def _pick_templates() -> dict[int, dict]:
    kinds = ["hline", "vline", "point", random.choice(KINDS)]
    random.shuffle(kinds)
    selected: dict[int, dict] = {}
    used_transforms: set[tuple[tuple[int, int], ...]] = set()
    for color, kind in zip(PAYLOAD_COLORS, kinds):
        candidates = [tpl for tpl in TEMPLATE_POOLS[kind] if tpl["transformed"] not in used_transforms]
        if not candidates:
            return {}
        template = random.choice(candidates)
        used_transforms.add(template["transformed"])
        selected[int(color)] = template
    return selected


def _validate_sample(grid: np.ndarray, selector_colors: dict[tuple[int, int], int], templates: dict[int, dict]) -> bool:
    try:
        correct = main(grid)
        copy_out = _solve(grid, mode="copy")
        alt_out = _solve(grid, mode="alt_axis")
        point_out = _solve(grid, mode="force_point")
    except Exception:
        return False

    if np.array_equal(correct, copy_out):
        return False
    if np.array_equal(correct, alt_out):
        return False
    if np.array_equal(correct, point_out):
        return False

    kinds = {tpl["kind"] for tpl in templates.values()}
    if kinds != {"hline", "vline", "point"} and len(kinds) < 3:
        return False

    expected = _render_output(
        {
            color: {
                "payload_color": color,
                "kind": templates[color]["kind"],
                "payload_deltas": templates[color]["payload_deltas"],
            }
            for color in PAYLOAD_COLORS
        },
        selector_colors,
        mode="correct",
    )
    return np.array_equal(correct, expected)


def _generate_once() -> np.ndarray | None:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    selector_box = _choose_selector_box(h, w)
    sprite_boxes = _place_sprite_boxes(h, w, selector_box, 4)
    if sprite_boxes is None:
        return None

    templates = _pick_templates()
    if len(templates) != 4:
        return None

    centers = list(MARKER_POSITIONS)
    colors = list(PAYLOAD_COLORS)
    random.shuffle(colors)
    selector_colors = {center: color for center, color in zip(centers, colors)}
    selector = _build_selector(selector_colors)

    grid = np.full((h, w), BG, dtype=int)
    sr0, sc0, _sr1, _sc1 = selector_box
    _paste(grid, selector, sr0, sc0)

    sprites = [_sprite_from_template(templates[color], color) for color in PAYLOAD_COLORS]
    random.shuffle(sprites)
    for sprite, (r, c) in zip(sprites, sprite_boxes):
        _paste(grid, sprite, r, c, transparent=BG)

    if not _validate_sample(grid, selector_colors, templates):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selector-mirror sprite puzzle")
