from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
RETRY_LIMIT = 800


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors(r: int, c: int, connectivity: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1
    if int(connectivity) == 8:
        yield r - 1, c - 1
        yield r - 1, c + 1
        yield r + 1, c - 1
        yield r + 1, c + 1


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for nr, nc in _neighbors(r, c, connectivity):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if seen[nr, nc] or not bool(mask[nr, nc]):
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))
        out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scale(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    assert k > 0, "scale factor must be positive"
    return np.kron(sprite, np.ones((k, k), dtype=sprite.dtype))


def _solid_full_lines(g: np.ndarray, bg: int) -> list[tuple[str, int]]:
    h, w = map(int, g.shape)
    lines: list[tuple[str, int]] = []
    for r in range(1, h - 1):
        color = int(g[r, 0])
        if color != bg and np.all(g[r, :] == color):
            lines.append(("row", r))
    for c in range(1, w - 1):
        color = int(g[0, c])
        if color != bg and np.all(g[:, c] == color):
            lines.append(("col", c))
    return lines


def _parse_templates(panel: np.ndarray, bg: int) -> dict[int, tuple[np.ndarray, tuple[int, int]]]:
    comps = _components(panel != bg, connectivity=8)
    assert len(comps) >= 2, "template panel must contain multiple sprites"

    templates: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        sprite = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = [int(c) for c in panel[coords[:, 0], coords[:, 1]] if int(c) != bg]
        counts = Counter(colors)
        assert len(counts) >= 2, "each sprite must be multicolor"

        min_count = min(counts.values())
        rarest = [color for color, count in counts.items() if count == min_count]
        assert len(rarest) == 1 and min_count == 1, "each sprite needs one singleton rarest color"
        anchor_color = int(rarest[0])
        assert anchor_color not in templates, "anchor colors must identify templates uniquely"

        anchor_cells = np.argwhere(sprite == anchor_color)
        assert anchor_cells.shape[0] == 1, "anchor color must occur once in its sprite crop"
        ar, ac = map(int, anchor_cells[0])
        templates[anchor_color] = (sprite, (ar, ac))

    return templates


def _single_sprite_has_anchor(sprite: np.ndarray, anchor_color: int) -> bool:
    comps = _components(sprite != BG, connectivity=8)
    if len(comps) != 1:
        return False
    colors = [int(c) for c in sprite[comps[0][:, 0], comps[0][:, 1]] if int(c) != BG]
    counts = Counter(colors)
    if len(counts) < 2:
        return False
    min_count = min(counts.values())
    rarest = [color for color, count in counts.items() if count == min_count]
    return len(rarest) == 1 and int(rarest[0]) == int(anchor_color) and min_count == 1


def _parse_markers(work: np.ndarray, bg: int, anchor_colors: set[int]) -> list[dict]:
    non_bg = {int(c) for c in np.unique(work) if int(c) != bg}
    assert non_bg and non_bg <= anchor_colors, "work panel may contain only anchor-colored markers"

    covered = np.zeros(work.shape, dtype=bool)
    markers: list[dict] = []
    for color in non_bg:
        for coords in _components(work == int(color), connectivity=4):
            r0, c0, r1, c1 = _bbox(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            assert height == width and height in (1, 2, 3), "markers must be 1x1, 2x2, or 3x3 squares"
            assert coords.shape[0] == height * width, "marker component must be solid"
            assert np.all(work[r0 : r1 + 1, c0 : c1 + 1] == int(color)), "marker bbox must be filled"
            covered[coords[:, 0], coords[:, 1]] = True
            markers.append({"color": int(color), "bbox": (r0, c0, r1, c1), "size": int(height)})

    assert np.array_equal(covered, work != bg), "all work foreground pixels must belong to markers"
    markers.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3]))
    return markers


def _stamp_work(work: np.ndarray, templates: dict[int, tuple[np.ndarray, tuple[int, int]]], bg: int) -> np.ndarray:
    markers = _parse_markers(work, bg, set(templates))
    cell_owner: dict[tuple[int, int], int] = {}
    for idx, marker in enumerate(markers):
        r0, c0, r1, c1 = marker["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                assert (r, c) not in cell_owner, "markers must not overlap"
                cell_owner[(r, c)] = idx

    painted: dict[tuple[int, int], int] = {}
    h, w = map(int, work.shape)
    for idx, marker in enumerate(markers):
        color = int(marker["color"])
        k = int(marker["size"])
        mr0, mc0, mr1, mc1 = marker["bbox"]
        sprite, (ar, ac) = templates[color]
        scaled = _scale(sprite, k)
        top = mr0 - ar * k
        left = mc0 - ac * k
        sh, sw = map(int, scaled.shape)
        assert 0 <= top and 0 <= left and top + sh <= h and left + sw <= w, "scaled sprite must fit in work panel"
        assert np.all(scaled[ar * k : (ar + 1) * k, ac * k : (ac + 1) * k] == color)

        for lr, lc in np.argwhere(scaled != bg):
            lr, lc = int(lr), int(lc)
            r, c = top + lr, left + lc
            owner = cell_owner.get((r, c))
            assert owner is None or owner == idx, "a stamp may cover only its own marker"
            if owner is None:
                assert int(work[r, c]) == bg, "stamp foreground must land on empty work background"
            assert (r, c) not in painted, "stamps must not overlap"
            painted[(r, c)] = int(scaled[lr, lc])

        for r in range(mr0, mr1 + 1):
            for c in range(mc0, mc1 + 1):
                assert painted.get((r, c)) == color, "anchor block must exactly cover the marker"

    out = work.copy()
    for (r, c), color in painted.items():
        out[r, c] = int(color)
    return out


def _solve_split(template: np.ndarray, work: np.ndarray, bg: int) -> np.ndarray:
    templates = _parse_templates(template, bg)
    return _stamp_work(work, templates, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each solid square marker with the matching sprite scaled by that marker's side length.

    Input structure:
    - A dominant background grid is split by one full separator line into a
      template panel and a work panel.
    - The template panel contains several separated multicolor sprites; in each
      sprite, the unique singleton rarest color pixel is its anchor.
    - The work panel contains separated solid square markers of side length 1,
      2, or 3, and each marker color equals one sprite's anchor color.

    Rule:
    - For every marker, select the template whose anchor color matches the
      marker color.
    - Scale that entire sprite by the marker side length, replacing each sprite
      pixel by a same-colored k by k block.
    - Stamp the scaled sprite into the work panel so the scaled anchor block
      exactly covers the marker square; then return only the work panel.

    Key insight:
    - The marker is a three-part key: its color chooses the sprite, its square
      location fixes the anchor alignment, and its side length fixes the scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    bg = _get_bg(g)

    solutions: list[np.ndarray] = []
    for kind, idx in _solid_full_lines(g, bg):
        if kind == "row":
            first, second = g[:idx, :], g[idx + 1 :, :]
        else:
            first, second = g[:, :idx], g[:, idx + 1 :]
        if first.size == 0 or second.size == 0:
            continue

        for template, work in ((first, second), (second, first)):
            try:
                solutions.append(_solve_split(template, work, bg))
            except (AssertionError, ValueError):
                continue

    assert len(solutions) == 1, "input must determine one template/work split"
    return solutions[0]


# --------------- generator ---------------


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(100):
        h = random.choice((2, 3, 3))
        w = random.choice((2, 3, 3))
        area = random.randint(4, min(7, h * w))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            options = []
            for r, c in cells:
                for nr, nc in _neighbors(r, c, 4):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        options.append((nr, nc))
            if not options:
                break
            cells.add(random.choice(options))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        r0, c0, r1, c1 = _bbox(np.argwhere(mask))
        crop = mask[r0 : r1 + 1, c0 : c1 + 1]
        if crop.shape[0] >= 2 and crop.shape[1] >= 2:
            return crop
    return None


def _build_sprite(anchor_color: int, body_palette: list[int]) -> dict | None:
    for _ in range(100):
        mask = _random_connected_mask()
        if mask is None:
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        if len(coords) < 4:
            continue
        anchor = random.choice(coords)
        body_cells = [cell for cell in coords if cell != anchor]
        possible_body_colors = min(2, len(body_palette), len(body_cells) // 2)
        if possible_body_colors < 1:
            continue
        body_count = 2 if possible_body_colors >= 2 and random.random() < 0.7 else 1
        body_colors = random.sample(body_palette, body_count)

        sprite = np.full(mask.shape, BG, dtype=int)
        sprite[anchor] = int(anchor_color)
        random.shuffle(body_cells)
        if body_count == 1:
            for r, c in body_cells:
                sprite[r, c] = int(body_colors[0])
        else:
            counts = [2] * body_count
            for _leftover in range(len(body_cells) - sum(counts)):
                counts[random.randrange(body_count)] += 1
            pos = 0
            for color, count in zip(body_colors, counts):
                for r, c in body_cells[pos : pos + count]:
                    sprite[r, c] = int(color)
                pos += count

        if _single_sprite_has_anchor(sprite, int(anchor_color)):
            return {"sprite": sprite, "anchor": anchor, "anchor_color": int(anchor_color)}
    return None


def _rect_clear(mask: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> bool:
    mh, mw = map(int, mask.shape)
    if r0 < 0 or c0 < 0 or r0 + h > mh or c0 + w > mw:
        return False
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(mh, r0 + h + margin)
    cc1 = min(mw, c0 + w + margin)
    return not bool(np.any(mask[rr0:rr1, cc0:cc1]))


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    fg = patch != BG
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[fg] = patch[fg]


def _place_templates(height: int, width: int, specs: list[dict]) -> np.ndarray | None:
    panel = np.full((height, width), BG, dtype=int)
    occupied = np.zeros(panel.shape, dtype=bool)
    for spec in specs:
        sprite = spec["sprite"]
        sh, sw = map(int, sprite.shape)
        rows = list(range(1, height - sh))
        cols = list(range(1, width - sw))
        random.shuffle(rows)
        random.shuffle(cols)
        placed = False
        for r0 in rows:
            for c0 in cols:
                if not _rect_clear(occupied, r0, c0, sh, sw, margin=1):
                    continue
                _paint_patch(panel, sprite, r0, c0)
                occupied[r0 : r0 + sh, c0 : c0 + sw] = True
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return panel


def _build_plans(sprite_count: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        marker_count = random.choice((4, 4, 5))
        sizes = [1, 2, 3]
        while len(sizes) < marker_count:
            sizes.append(random.choice((1, 2, 3)))
        random.shuffle(sizes)
        owners = list(range(sprite_count))
        while len(owners) < marker_count:
            owners.append(random.randrange(sprite_count))
        random.shuffle(owners)
        plans = [(int(owner), int(k)) for owner, k in zip(owners, sizes)]
        used = {idx: set() for idx in range(sprite_count)}
        for owner, k in plans:
            used[owner].add(k)
        if set(k for _owner, k in plans) == {1, 2, 3} and any(len(v) >= 2 for v in used.values()):
            return plans
    return None


def _place_work(height: int, width: int, specs: list[dict], plans: list[tuple[int, int]]):
    work = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(
        range(len(plans)),
        key=lambda i: specs[plans[i][0]]["sprite"].shape[0] * specs[plans[i][0]]["sprite"].shape[1] * plans[i][1] ** 2,
        reverse=True,
    )

    for plan_index in order:
        sprite_idx, k = plans[plan_index]
        spec = specs[sprite_idx]
        sprite = spec["sprite"]
        anchor_r, anchor_c = spec["anchor"]
        scaled = _scale(sprite, k)
        sh, sw = map(int, scaled.shape)
        if sh > height or sw > width:
            return None

        placed = False
        for _ in range(250):
            top = random.randint(0, height - sh)
            left = random.randint(0, width - sw)
            if not _rect_clear(occupied, top, left, sh, sw, margin=1):
                continue
            mr0 = top + anchor_r * k
            mc0 = left + anchor_c * k
            if not np.all(work[mr0 : mr0 + k, mc0 : mc0 + k] == BG):
                continue

            work[mr0 : mr0 + k, mc0 : mc0 + k] = int(spec["anchor_color"])
            _paint_patch(expected, scaled, top, left)
            occupied[top : top + sh, left : left + sw] = True
            placed = True
            break
        if not placed:
            return None

    return work, expected


def _has_only_one_separator(grid: np.ndarray, separator_row: int) -> bool:
    bg = _get_bg(grid)
    h, w = map(int, grid.shape)
    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r, :] == color) and r != separator_row:
            return False
    for c in range(w):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color):
            return False
    return True


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    specs: list[dict],
    plans: list[tuple[int, int]],
    template_height: int,
) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not _has_only_one_separator(grid, template_height):
        return False
    if set(k for _owner, k in plans) != {1, 2, 3}:
        return False

    used = {idx: set() for idx in range(len(specs))}
    for owner, k in plans:
        used[owner].add(k)
    if any(not sizes for sizes in used.values()):
        return False
    if not any(len(sizes) >= 2 for sizes in used.values()):
        return False
    if len({spec["anchor"] for spec in specs}) < min(2, len(specs)):
        return False

    work = grid[template_height + 1 :, :]
    marker_area = sum(k * k for _owner, k in plans)
    if int(np.count_nonzero(work != BG)) != marker_area:
        return False
    output_area = int(np.count_nonzero(expected != BG))
    if output_area <= marker_area + 2 * len(plans):
        return False
    if output_area > int(expected.size * 0.55):
        return False

    for spec in specs:
        colors = {int(c) for c in np.unique(spec["sprite"]) if int(c) not in (BG, spec["anchor_color"])}
        if not colors:
            return False
        if not any(bool(np.any(expected == color)) for color in colors):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.choice((2, 3, 3))
    separator_color = random.choice(COLORS)
    anchor_colors = random.sample([c for c in COLORS if c != separator_color], sprite_count)
    body_palette = [c for c in COLORS if c not in set(anchor_colors + [separator_color])]
    if len(body_palette) < 2:
        return None

    specs: list[dict] = []
    for color in anchor_colors:
        spec = _build_sprite(color, body_palette)
        if spec is None:
            return None
        specs.append(spec)

    width = random.randint(18, 30)
    template_height = random.randint(7, 10)
    max_work_height = 30 - template_height - 1
    if max_work_height < 13:
        return None
    work_height = random.randint(13, max_work_height)

    template = _place_templates(template_height, width, specs)
    if template is None:
        return None
    plans = _build_plans(sprite_count)
    if plans is None:
        return None

    placed = _place_work(work_height, width, specs, plans)
    if placed is None:
        return None
    work, expected = placed

    grid = np.vstack([template, np.full((1, width), int(separator_color), dtype=int), work])
    if not _semantic_checks(grid, expected, specs, plans, template_height):
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/4/001/6")
