from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BODY_COLORS = (1, 2)
MARKER_COLORS = (3, 4)
TARGET_COLORS = (6, 7)
TEMPLATE_COLOR = 8
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_ROT = {"up": 0, "right": 3, "down": 2, "left": 1}
CORNER_DIRS = {
    "tl": ("right", "down"),
    "tr": ("left", "down"),
    "bl": ("right", "up"),
    "br": ("left", "up"),
}
TEMPLATE_LIBRARY = [
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
]
MAX_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    parts: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        parts.append(np.array(cells, dtype=int))
    return parts


def _find_multicolor_components(grid: np.ndarray, colors: set[int]) -> list[np.ndarray]:
    return _components(np.isin(grid, list(colors)))


def _mask_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _rot_signature(mask: np.ndarray) -> tuple:
    rots = []
    cur = mask.copy()
    for _ in range(4):
        rots.append((cur.shape, tuple(map(tuple, np.argwhere(cur)))))
        cur = np.rot90(cur)
    return min(rots)


def _same_up_to_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    return _rot_signature(a) == _rot_signature(b)


def _rotate_mask(mask: np.ndarray, side: str) -> np.ndarray:
    return np.rot90(mask, SIDE_TO_ROT[side]).copy()


def _marker_side(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = marker
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "up"
    if r == r1:
        return "down"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("marker must lie on a bbox side")


def _parse_rules(grid: np.ndarray) -> dict[int, dict]:
    teal_parts = _components(grid == TEMPLATE_COLOR)
    teal_info = []
    for coords in teal_parts:
        teal_info.append({"coords": coords, "bbox": _bbox(coords), "mask": _crop_mask(coords)})

    rules: dict[int, dict] = {}
    used_templates: set[int] = set()
    for sr, sc in np.argwhere(np.isin(grid, MARKER_COLORS)):
        sr = int(sr)
        sc = int(sc)
        src_color = int(grid[sr, sc])
        neighbors = []
        for dr, dc in DIR4:
            tr, tc = sr + dr, sc + dc
            if _in_bounds(grid.shape, tr, tc) and int(grid[tr, tc]) in TARGET_COLORS:
                neighbors.append((tr, tc))
        if not neighbors:
            continue
        assert len(neighbors) == 1, "rule source pixel must touch exactly one target pixel"
        tr, tc = neighbors[0]
        direction = (tr - sr, tc - sc)

        candidates: list[tuple[int, int]] = []
        for idx, info in enumerate(teal_info):
            r0, c0, r1, c1 = info["bbox"]
            gap = None
            if direction == (0, 1) and c0 > tc and r0 <= sr <= r1:
                gap = c0 - tc - 1
            elif direction == (0, -1) and c1 < tc and r0 <= sr <= r1:
                gap = tc - c1 - 1
            elif direction == (1, 0) and r0 > tr and c0 <= sc <= c1:
                gap = r0 - tr - 1
            elif direction == (-1, 0) and r1 < tr and c0 <= sc <= c1:
                gap = tr - r1 - 1
            if gap is not None and gap >= 1:
                candidates.append((gap, idx))
        assert candidates, "each rule pair must point at a teal template"
        candidates.sort()
        best_gap = candidates[0][0]
        best = [idx for gap, idx in candidates if gap == best_gap]
        assert len(best) == 1, "nearest forward teal template must be unique"
        template_idx = best[0]
        assert template_idx not in used_templates, "one template cannot serve two rules"
        used_templates.add(template_idx)
        rules[src_color] = {
            "source_color": src_color,
            "target_color": int(grid[tr, tc]),
            "pair": ((sr, sc), (tr, tc)),
            "template_coords": teal_info[template_idx]["coords"],
            "template": teal_info[template_idx]["mask"],
        }

    assert len(rules) == len(teal_info) == 2, "expected exactly two rule examples"
    assert set(rules) == set(MARKER_COLORS), "both marker colors must have rules"
    return rules


def _parse_sprites(grid: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for coords in _find_multicolor_components(grid, set(BODY_COLORS) | set(MARKER_COLORS)):
        colors = {int(grid[r, c]) for r, c in coords}
        body_colors = [color for color in BODY_COLORS if color in colors]
        marker_cells = [(int(r), int(c)) for r, c in coords if int(grid[r, c]) in MARKER_COLORS]
        if len(body_colors) != 1 or len(marker_cells) != 1:
            continue
        if colors - {body_colors[0], int(grid[marker_cells[0]])}:
            continue
        marker = marker_cells[0]
        info = {
            "coords": coords,
            "bbox": _bbox(coords),
            "body_color": body_colors[0],
            "marker_color": int(grid[marker]),
            "marker_pos": marker,
        }
        info["side"] = _marker_side(info["marker_pos"], info["bbox"])
        sprites.append(info)
    assert 2 <= len(sprites) <= 3, "expected 2-3 marked sprites"
    return sprites


def _apply_rules(grid: np.ndarray, rules: dict[int, dict], sprites: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for rule in rules.values():
        for r, c in rule["pair"]:
            out[r, c] = BG
        for r, c in rule["template_coords"]:
            out[int(r), int(c)] = BG

    for sprite in sprites:
        rule = rules[sprite["marker_color"]]
        stamp = _rotate_mask(rule["template"], sprite["side"])
        r0, c0 = sprite["marker_pos"]
        h, w = stamp.shape
        assert r0 + h <= out.shape[0] and c0 + w <= out.shape[1], "stamp must fit the canvas"
        out[r0 : r0 + h, c0 : c0 + w][stamp] = rule["target_color"]

    for sprite in sprites:
        r, c = sprite["marker_pos"]
        out[r, c] = sprite["body_color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the teal stamp rule from each sprite's embedded marker color and rotate it by the marker-side direction.

    Input structure:
    - The grid has black background and two separated rule examples in the corners.
    - Each rule example is a source-target adjacent pair whose source color is green or yellow,
      followed farther along the same direction by one teal connected template.
    - The grid also contains 2-3 multicolor sprites: one red or blue body plus one embedded
      green or yellow marker pixel on the sprite bbox edge.

    Rule:
    - Parse each rule example into a source color, a target color, and the nearest forward teal template.
    - For each sprite, match its embedded marker color to the rule source color.
    - Read the marker side from the sprite bbox with row priority: top, bottom, left, then right.
    - Rotate the matched template from its default up orientation to that side, anchor the rotated
      template's top-left corner at the marker position, and stamp it in the rule's target color.
    - Erase both rule examples and restore every embedded marker pixel back to the sprite body color.

    Key insight:
    - The marker color chooses which rule to use, while the marker's side inside the sprite chooses
      how the matched template must be rotated before stamping.
    """
    g = np.asarray(grid, dtype=int)
    rules = _parse_rules(g)
    sprites = _parse_sprites(g)
    return _apply_rules(g, rules, sprites)


def _separated(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], gap: int) -> bool:
    for r, c in cells:
        for rr, cc in occupied:
            if max(abs(r - rr), abs(c - cc)) <= gap:
                return False
    return True


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _random_polyomino(area: int) -> set[tuple[int, int]] | None:
    candidates = [(h, w) for h in range(2, 5) for w in range(2, 5) if h * w >= area]
    random.shuffle(candidates)
    for h, w in candidates:
        for _ in range(40):
            start = (random.randrange(h), random.randrange(w))
            cells = {start}
            while len(cells) < area:
                frontier = []
                for r, c in cells:
                    for dr, dc in DIR4:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                cells.add(random.choice(frontier))
            if len(cells) != area:
                continue
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            if max(rows) == min(rows) or max(cols) == min(cols):
                continue
            if len(cells) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1):
                continue
            r0 = min(rows)
            c0 = min(cols)
            return {(r - r0, c - c0) for r, c in cells}
    return None


def _make_sprite_patch(
    body_color: int,
    marker_color: int,
    side: str,
    body_area: int,
    prefer_corner: bool,
) -> dict | None:
    for _ in range(150):
        body = _random_polyomino(body_area)
        if body is None:
            continue
        rows = [r for r, _ in body]
        cols = [c for _, c in body]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        if side == "up":
            candidates = [(r, c) for r, c in body if r == min_r and (r - 1, c) not in body]
            if prefer_corner:
                edge = [p for p in candidates if p[1] in (min_c, max_c)]
                if edge:
                    candidates = edge
            elif max_c - min_c >= 2:
                inner = [p for p in candidates if p[1] not in (min_c, max_c)]
                if inner:
                    candidates = inner
            marker_builder = lambda rc: (rc[0] - 1, rc[1])
        elif side == "down":
            candidates = [(r, c) for r, c in body if r == max_r and (r + 1, c) not in body]
            if prefer_corner:
                edge = [p for p in candidates if p[1] in (min_c, max_c)]
                if edge:
                    candidates = edge
            elif max_c - min_c >= 2:
                inner = [p for p in candidates if p[1] not in (min_c, max_c)]
                if inner:
                    candidates = inner
            marker_builder = lambda rc: (rc[0] + 1, rc[1])
        elif side == "left":
            candidates = [(r, c) for r, c in body if c == min_c and (r, c - 1) not in body]
            if max_r - min_r >= 2:
                inner = [p for p in candidates if p[0] not in (min_r, max_r)]
                if inner:
                    candidates = inner
            marker_builder = lambda rc: (rc[0], rc[1] - 1)
        else:
            candidates = [(r, c) for r, c in body if c == max_c and (r, c + 1) not in body]
            if max_r - min_r >= 2:
                inner = [p for p in candidates if p[0] not in (min_r, max_r)]
                if inner:
                    candidates = inner
            marker_builder = lambda rc: (rc[0], rc[1] + 1)

        random.shuffle(candidates)
        for attach in candidates:
            marker = marker_builder(attach)
            full = set(body)
            full.add(marker)
            r0 = min(r for r, _ in full)
            c0 = min(c for _, c in full)
            norm_full = {(r - r0, c - c0) for r, c in full}
            norm_body = {(r - r0, c - c0) for r, c in body}
            norm_marker = (marker[0] - r0, marker[1] - c0)
            full_coords = np.array(sorted(norm_full), dtype=int)
            bbox = _bbox(full_coords)
            if _marker_side(norm_marker, bbox) != side:
                continue
            h = max(r for r, _ in norm_full) + 1
            w = max(c for _, c in norm_full) + 1
            patch = np.zeros((h, w), dtype=int)
            for r, c in norm_body:
                patch[r, c] = body_color
            patch[norm_marker] = marker_color
            return {
                "patch": patch,
                "marker_pos": norm_marker,
                "body_color": body_color,
                "marker_color": marker_color,
                "side": side,
                "coords": norm_full,
            }
    return None


def _rule_source_ranges(shape: tuple[int, int], corner: str) -> tuple[range, range]:
    h, w = shape
    if corner == "tl":
        return range(1, min(4, h - 1)), range(1, min(4, w - 1))
    if corner == "tr":
        return range(1, min(4, h - 1)), range(max(1, w - 4), w - 1)
    if corner == "bl":
        return range(max(1, h - 4), h - 1), range(1, min(4, w - 1))
    return range(max(1, h - 4), h - 1), range(max(1, w - 4), w - 1)


def _enumerate_rule_placements(
    shape: tuple[int, int], corner: str, template: np.ndarray
) -> list[dict]:
    h, w = template.shape
    rows, cols = _rule_source_ranges(shape, corner)
    out = []
    for side in CORNER_DIRS[corner]:
        for sr in rows:
            for sc in cols:
                if side == "right":
                    tr, tc = sr, sc + 1
                    for gap in (1, 2):
                        for offset in range(h):
                            top = sr - offset
                            left = tc + 1 + gap
                            cells = {
                                (top + r, left + c)
                                for r, c in np.argwhere(template)
                            }
                            pair = {(sr, sc), (tr, tc)}
                            if all(_in_bounds(shape, r, c) for r, c in cells | pair):
                                out.append({"side": side, "pair": pair, "template": cells})
                elif side == "left":
                    tr, tc = sr, sc - 1
                    for gap in (1, 2):
                        for offset in range(h):
                            top = sr - offset
                            left = tc - gap - w
                            cells = {
                                (top + r, left + c)
                                for r, c in np.argwhere(template)
                            }
                            pair = {(sr, sc), (tr, tc)}
                            if all(_in_bounds(shape, r, c) for r, c in cells | pair):
                                out.append({"side": side, "pair": pair, "template": cells})
                elif side == "down":
                    tr, tc = sr + 1, sc
                    for gap in (1, 2):
                        for offset in range(w):
                            top = tr + 1 + gap
                            left = sc - offset
                            cells = {
                                (top + r, left + c)
                                for r, c in np.argwhere(template)
                            }
                            pair = {(sr, sc), (tr, tc)}
                            if all(_in_bounds(shape, r, c) for r, c in cells | pair):
                                out.append({"side": side, "pair": pair, "template": cells})
                else:
                    tr, tc = sr - 1, sc
                    for gap in (1, 2):
                        for offset in range(w):
                            top = tr - gap - h
                            left = sc - offset
                            cells = {
                                (top + r, left + c)
                                for r, c in np.argwhere(template)
                            }
                            pair = {(sr, sc), (tr, tc)}
                            if all(_in_bounds(shape, r, c) for r, c in cells | pair):
                                out.append({"side": side, "pair": pair, "template": cells})
    random.shuffle(out)
    return out


def _stamp_cells(relative_marker: tuple[int, int], stamp: np.ndarray) -> set[tuple[int, int]]:
    mr, mc = relative_marker
    return {(mr + int(r), mc + int(c)) for r, c in np.argwhere(stamp)}


def _place_sprite(
    shape: tuple[int, int],
    sprite: dict,
    rule_template: np.ndarray,
    occupied_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    forbidden_output: set[tuple[int, int]],
) -> dict | None:
    patch = sprite["patch"]
    ph, pw = patch.shape
    stamp = _rotate_mask(rule_template, sprite["side"])
    stamp_rel = _stamp_cells(sprite["marker_pos"], stamp)
    final_rel = set(sprite["coords"]) | stamp_rel
    if not (stamp_rel - set(sprite["coords"])):
        return None
    max_r = max(r for r, _ in final_rel)
    max_c = max(c for _, c in final_rel)

    candidates = []
    for r0 in range(shape[0] - max(max_r + 1, ph) + 1):
        for c0 in range(shape[1] - max(max_c + 1, pw) + 1):
            sprite_cells = {
                (r0 + int(r), c0 + int(c))
                for r, c in np.argwhere(patch != BG)
            }
            final_cells = {(r0 + r, c0 + c) for r, c in final_rel}
            if not _separated(sprite_cells, occupied_input, gap=1):
                continue
            if not _separated(final_cells, occupied_output, gap=1):
                continue
            if not _separated(final_cells, forbidden_output, gap=1):
                continue
            candidates.append((r0, c0, sprite_cells, final_cells, stamp_rel))
    if not candidates:
        return None
    r0, c0, sprite_cells, final_cells, stamp_rel = random.choice(candidates)
    placed = dict(sprite)
    placed["top_left"] = (r0, c0)
    placed["input_cells"] = sprite_cells
    placed["final_cells"] = final_cells
    placed["stamp_cells"] = {(r0 + r, c0 + c) for r, c in stamp_rel}
    placed["marker_global"] = (r0 + sprite["marker_pos"][0], c0 + sprite["marker_pos"][1])
    return placed


def _sample_templates() -> tuple[np.ndarray, np.ndarray]:
    first = random.choice(TEMPLATE_LIBRARY)
    others = [tpl for tpl in TEMPLATE_LIBRARY if not _same_up_to_rotation(first, tpl)]
    second = random.choice(others)
    return first.copy(), second.copy()


def _build_expected_from_meta(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = np.array(grid, copy=True)
    for rule in meta["rules"].values():
        _paint_cells(out, rule["pair_cells"], BG)
        _paint_cells(out, rule["template_cells"], BG)
    for sprite in meta["sprites"]:
        stamp = _rotate_mask(meta["rules"][sprite["marker_color"]]["template"], sprite["side"])
        mr, mc = sprite["marker_global"]
        out[mr : mr + stamp.shape[0], mc : mc + stamp.shape[1]][stamp] = meta["rules"][sprite["marker_color"]][
            "target_color"
        ]
    for sprite in meta["sprites"]:
        mr, mc = sprite["marker_global"]
        out[mr, mc] = sprite["body_color"]
    return out


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    rules = _parse_rules(grid)
    sprites = _parse_sprites(grid)
    if len(sprites) != len(meta["sprites"]):
        return False
    for color in MARKER_COLORS:
        if color not in rules or color not in meta["rules"]:
            return False
        if rules[color]["target_color"] != meta["rules"][color]["target_color"]:
            return False
        if not _mask_equal(rules[color]["template"], meta["rules"][color]["template"]):
            return False

    by_marker = {tuple(sprite["marker_pos"]): sprite for sprite in sprites}
    for sprite in meta["sprites"]:
        inferred = by_marker.get(sprite["marker_global"])
        if inferred is None:
            return False
        if inferred["marker_color"] != sprite["marker_color"] or inferred["side"] != sprite["side"]:
            return False

    out = main(grid)
    expected = _build_expected_from_meta(grid, meta)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, grid):
        return False

    sprite_marker_colors = [sprite["marker_color"] for sprite in meta["sprites"]]
    if set(sprite_marker_colors) != set(MARKER_COLORS):
        return False
    if all(sprite["side"] == "up" for sprite in meta["sprites"]):
        return False

    for sprite in meta["sprites"]:
        template = meta["rules"][sprite["marker_color"]]["template"]
        if sprite["side"] != "up" and _mask_equal(_rotate_mask(template, sprite["side"]), template):
            return False
        other_color = 3 if sprite["marker_color"] == 4 else 4
        other_rule = meta["rules"][other_color]
        same_shape = _mask_equal(
            _rotate_mask(template, sprite["side"]),
            _rotate_mask(other_rule["template"], sprite["side"]),
        )
        if same_shape and meta["rules"][sprite["marker_color"]]["target_color"] == other_rule["target_color"]:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        shape = (random.randint(16, 22), random.randint(16, 22))
        grid = np.zeros(shape, dtype=int)
        template_a, template_b = _sample_templates()
        target_a, target_b = random.sample(TARGET_COLORS, 2)
        corners = random.sample(list(CORNER_DIRS), 2)
        rule_meta = {
            3: {"template": template_a, "target_color": target_a},
            4: {"template": template_b, "target_color": target_b},
        }

        occupied_input: set[tuple[int, int]] = set()
        used_rule_cells: set[tuple[int, int]] = set()
        ok = True
        for src_color, corner in zip(MARKER_COLORS, corners):
            placed = None
            placements = _enumerate_rule_placements(shape, corner, rule_meta[src_color]["template"])
            for choice in placements:
                all_cells = choice["pair"] | choice["template"]
                if not _separated(all_cells, occupied_input, gap=1):
                    continue
                placed = choice
                break
            if placed is None:
                ok = False
                break
            # Recover source/target from placement side.
            if placed["side"] == "right":
                src = min(placed["pair"], key=lambda rc: rc[1])
                dst = max(placed["pair"], key=lambda rc: rc[1])
            elif placed["side"] == "left":
                src = max(placed["pair"], key=lambda rc: rc[1])
                dst = min(placed["pair"], key=lambda rc: rc[1])
            elif placed["side"] == "down":
                src = min(placed["pair"], key=lambda rc: rc[0])
                dst = max(placed["pair"], key=lambda rc: rc[0])
            else:
                src = max(placed["pair"], key=lambda rc: rc[0])
                dst = min(placed["pair"], key=lambda rc: rc[0])

            grid[src] = src_color
            grid[dst] = rule_meta[src_color]["target_color"]
            _paint_cells(grid, placed["template"], TEMPLATE_COLOR)
            occupied_input |= all_cells
            used_rule_cells |= all_cells
            rule_meta[src_color]["pair_cells"] = placed["pair"]
            rule_meta[src_color]["template_cells"] = placed["template"]
            rule_meta[src_color]["corner"] = corner
            rule_meta[src_color]["rule_side"] = placed["side"]
        if not ok:
            continue

        n_sprites = random.randint(2, 3)
        marker_colors = [3, 4] + ([random.choice(MARKER_COLORS)] if n_sprites == 3 else [])
        random.shuffle(marker_colors)
        sides = random.sample(["up", "right", "down", "left"], n_sprites)
        occupied_output: set[tuple[int, int]] = set()
        sprites_meta = []

        for idx in range(n_sprites):
            marker_color = marker_colors[idx]
            body_color = random.choice(BODY_COLORS)
            side = sides[idx]
            prefer_corner = side in {"up", "down"} and random.random() < 0.35
            sprite = _make_sprite_patch(
                body_color=body_color,
                marker_color=marker_color,
                side=side,
                body_area=random.randint(4, 8),
                prefer_corner=prefer_corner,
            )
            if sprite is None:
                ok = False
                break
            placed = _place_sprite(
                shape=shape,
                sprite=sprite,
                rule_template=rule_meta[marker_color]["template"],
                occupied_input=occupied_input,
                occupied_output=occupied_output,
                forbidden_output=used_rule_cells,
            )
            if placed is None:
                ok = False
                break
            rr, cc = placed["top_left"]
            patch = placed["patch"]
            mask = patch != BG
            grid[rr : rr + patch.shape[0], cc : cc + patch.shape[1]][mask] = patch[mask]
            occupied_input |= placed["input_cells"]
            occupied_output |= placed["final_cells"]
            sprites_meta.append(placed)
        if not ok:
            continue

        meta = {"rules": rule_meta, "sprites": sprites_meta}
        if _validate_case(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
