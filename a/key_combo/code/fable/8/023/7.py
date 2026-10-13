from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_GEN_ATTEMPTS = 400


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "shape mask must be non-empty"
    r0, c0, r1, c1 = bbox_coords(coords)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1].astype(np.uint8)
    return tuple(map(int, cropped.shape)), cropped.tobytes()


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _center_distance(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ra, ca = _bbox_center(box_a)
    rb, cb = _bbox_center(box_b)
    return abs(ra - rb) + abs(ca - cb)


def _enclosed_holes(occupied: np.ndarray) -> list[dict]:
    bg_mask = ~occupied
    comps = components_mask(bg_mask, connectivity=4)
    holes: list[dict] = []
    h, w = occupied.shape
    for coords in comps:
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        ):
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        holes.append({"bbox_local": (r0, c0, r1, c1), "mask": mask, "sig": _mask_signature(mask)})
    return holes


def _piece_from_sprite(component_id: str, sprite: np.ndarray, box: tuple[int, int, int, int], depth: int = 0) -> dict:
    return {
        "id": component_id,
        "sprite": sprite.copy(),
        "bbox": tuple(map(int, box)),
        "shape_sig": _mask_signature(sprite != BG),
        "source_ids": [component_id],
        "depth": int(depth),
    }


def _parse_input(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    bg = get_bg(grid)
    objects = sorted(
        find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"),
        key=lambda obj: obj["bbox"],
    )
    pieces: list[dict] = []
    shells: list[dict] = []
    for index, obj in enumerate(objects):
        box = tuple(map(int, obj["bbox"]))
        sprite = crop_bbox(grid, box)
        holes = _enclosed_holes(sprite != bg)
        if len(holes) == 0:
            pieces.append(_piece_from_sprite(f"obj{index}", sprite, box, depth=0))
            continue
        assert len(holes) == 1, "each hollow object must contain exactly one enclosed hole"
        hole = holes[0]
        shells.append(
            {
                "id": f"obj{index}",
                "sprite": sprite.copy(),
                "bbox": box,
                "hole_bbox_local": hole["bbox_local"],
                "hole_sig": hole["sig"],
            }
        )
    return pieces, shells


def _assemble_piece(piece: dict, shell: dict, new_id: str) -> dict:
    assembled = shell["sprite"].copy()
    hole_r0, hole_c0, _hole_r1, _hole_c1 = shell["hole_bbox_local"]
    paste(assembled, piece["sprite"], hole_r0, hole_c0, transparent=BG)
    return {
        "id": new_id,
        "sprite": assembled,
        "bbox": shell["bbox"],
        "shape_sig": _mask_signature(assembled != BG),
        "source_ids": piece["source_ids"] + [shell["id"]],
        "depth": int(piece["depth"]) + 1,
    }


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    pieces, shells = _parse_input(g)
    assert pieces and shells, "expected both solid pieces and hollow shells"

    links: list[tuple[str, str]] = []
    step = 0
    while True:
        exact = [(si, pi) for si, shell in enumerate(shells) for pi, piece in enumerate(pieces) if shell["hole_sig"] == piece["shape_sig"]]
        if not exact:
            break
        shell_counts = Counter(si for si, _pi in exact)
        piece_counts = Counter(pi for _si, pi in exact)
        assert all(count == 1 for count in shell_counts.values()), "hole shape must identify a unique incoming piece"
        assert all(count == 1 for count in piece_counts.values()), "one piece cannot fit multiple holes"

        used_shells: set[int] = set()
        used_pieces: set[int] = set()
        new_pieces: list[dict] = []
        for si, pi in sorted(exact, key=lambda pair: shells[pair[0]]["bbox"]):
            shell = shells[si]
            piece = pieces[pi]
            links.append((piece["id"], shell["id"]))
            new_pieces.append(_assemble_piece(piece, shell, f"assembled_{step}_{si}_{pi}"))
            used_shells.add(si)
            used_pieces.add(pi)

        pieces = [piece for idx, piece in enumerate(pieces) if idx not in used_pieces] + new_pieces
        shells = [shell for idx, shell in enumerate(shells) if idx not in used_shells]
        step += 1

    out = np.full_like(g, bg)
    for shell in shells:
        r0, c0, _r1, _c1 = shell["bbox"]
        paste(out, shell["sprite"], r0, c0, transparent=bg)
    for piece in pieces:
        r0, c0, _r1, _c1 = piece["bbox"]
        paste(out, piece["sprite"], r0, c0, transparent=bg)

    return out, {"roots": pieces, "remaining_shells": shells, "links": links}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recursively insert each solid key into its matching hole, then use the filled lock as the next key.

    Input structure:
    - A black background contains separated monochrome objects.
    - Some objects are solid sprites with no enclosed hole; others are hollow shells with exactly one enclosed black hole.
    - Each usable solid sprite matches one shell hole by exact translation-only shape equality, and a filled shell may itself become the next matching key.

    Rule:
    - Move every solid sprite into the uniquely matching shell hole, preserving the sprite's color.
    - After a shell is filled, treat the whole filled object as one new key whose occupied shape is its filled silhouette.
    - Repeat the same shape-to-hole assembly recursively until no further exact match remains.
    - Erase every object that moved away from its original position and keep only the final assembled root at the outer shell's original location; untouched distractors stay unchanged.

    Key insight:
    - Matching depends on positive-versus-negative shape equality inside the correct local coordinate frame, not on distance, and the next key is the filled silhouette of the previous shell.
    """
    return _solve(grid)[0]


def _random_connected_mask(height_range: tuple[int, int], width_range: tuple[int, int], *, require_irregular: bool) -> np.ndarray:
    for _ in range(200):
        h = random.randint(*height_range)
        w = random.randint(*width_range)
        total = h * w
        min_area = max(3, (total + 1) // 2)
        max_area = total - 1 if require_irregular else total
        if max_area < min_area:
            continue
        target_area = random.randint(min_area, max_area)
        start = (random.randrange(h), random.randrange(w))
        coords = {start}
        frontier = {start}
        while len(coords) < target_area:
            seed_r, seed_c = random.choice(tuple(frontier))
            options = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr = seed_r + dr
                cc = seed_c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in coords:
                    options.append((rr, cc))
            if not options:
                frontier.remove((seed_r, seed_c))
                if not frontier:
                    break
                continue
            cell = random.choice(options)
            coords.add(cell)
            frontier.add(cell)
        if len(coords) != target_area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in coords:
            mask[rr, cc] = True
        r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if require_irregular and mask.all():
            continue
        if len(_enclosed_holes(mask)) != 0:
            continue
        return mask
    raise RuntimeError("failed to sample a connected mask")


def _make_colored_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _build_shell_around(hole_mask: np.ndarray, color: int, reserved_sigs: set[tuple[tuple[int, int], bytes]]) -> tuple[np.ndarray, np.ndarray]:
    hole_h, hole_w = map(int, hole_mask.shape)
    for _ in range(300):
        margin_top = random.randint(1, 2)
        margin_bottom = random.randint(1, 2)
        margin_left = random.randint(1, 2)
        margin_right = random.randint(1, 2)
        core_h = hole_h + margin_top + margin_bottom
        core_w = hole_w + margin_left + margin_right

        extra_top = random.randint(0, 1)
        extra_bottom = random.randint(0, 1)
        extra_left = random.randint(0, 1)
        extra_right = random.randint(0, 1)
        total_h = core_h + extra_top + extra_bottom
        total_w = core_w + extra_left + extra_right
        if total_h > 19 or total_w > 19:
            continue

        filled_mask = np.zeros((total_h, total_w), dtype=bool)
        core_r0 = extra_top
        core_c0 = extra_left
        filled_mask[core_r0 : core_r0 + core_h, core_c0 : core_c0 + core_w] = True

        side_order = ["top", "bottom", "left", "right"]
        random.shuffle(side_order)
        for side in side_order:
            if side == "top" and extra_top:
                thickness = random.randint(1, extra_top)
                width = random.randint(1, core_w)
                start = random.randint(0, core_w - width)
                filled_mask[core_r0 - thickness : core_r0, core_c0 + start : core_c0 + start + width] = True
            elif side == "bottom" and extra_bottom:
                thickness = random.randint(1, extra_bottom)
                width = random.randint(1, core_w)
                start = random.randint(0, core_w - width)
                filled_mask[core_r0 + core_h : core_r0 + core_h + thickness, core_c0 + start : core_c0 + start + width] = True
            elif side == "left" and extra_left:
                thickness = random.randint(1, extra_left)
                height = random.randint(1, core_h)
                start = random.randint(0, core_h - height)
                filled_mask[core_r0 + start : core_r0 + start + height, core_c0 - thickness : core_c0] = True
            elif side == "right" and extra_right:
                thickness = random.randint(1, extra_right)
                height = random.randint(1, core_h)
                start = random.randint(0, core_h - height)
                filled_mask[core_r0 + start : core_r0 + start + height, core_c0 + core_w : core_c0 + core_w + thickness] = True

        filled_sig = _mask_signature(filled_mask)
        if filled_sig in reserved_sigs:
            continue

        shell_mask = filled_mask.copy()
        hole_r0 = core_r0 + margin_top
        hole_c0 = core_c0 + margin_left
        shell_mask[hole_r0 : hole_r0 + hole_h, hole_c0 : hole_c0 + hole_w][hole_mask] = False
        if len(_enclosed_holes(shell_mask)) != 1:
            continue
        return _make_colored_sprite(shell_mask, color), filled_mask
    raise RuntimeError("failed to build shell")


def _make_unique_solid_sprite(color: int, reserved_sigs: set[tuple[tuple[int, int], bytes]]) -> np.ndarray:
    for _ in range(200):
        mask = _random_connected_mask((2, 4), (2, 4), require_irregular=True)
        sig = _mask_signature(mask)
        if sig not in reserved_sigs:
            return _make_colored_sprite(mask, color)
    raise RuntimeError("failed to build unique solid distractor")


def _make_unique_shell(color: int, reserved_sigs: set[tuple[tuple[int, int], bytes]]) -> np.ndarray:
    for _ in range(200):
        hole_mask = _random_connected_mask((2, 3), (2, 4), require_irregular=True)
        hole_sig = _mask_signature(hole_mask)
        if hole_sig in reserved_sigs:
            continue
        sprite, filled_mask = _build_shell_around(hole_mask, color, reserved_sigs | {hole_sig})
        filled_sig = _mask_signature(filled_mask)
        if filled_sig in reserved_sigs:
            continue
        reserved_sigs.add(hole_sig)
        reserved_sigs.add(filled_sig)
        return sprite
    raise RuntimeError("failed to build unique shell distractor")


def _build_chain(depth: int, colors: list[int], reserved_sigs: set[tuple[tuple[int, int], bytes]]) -> tuple[list[dict], np.ndarray]:
    base_mask = _random_connected_mask((2, 4), (2, 4), require_irregular=True)
    current_sprite = _make_colored_sprite(base_mask, colors[0])
    current_mask = current_sprite != BG
    reserved_sigs.add(_mask_signature(current_mask))

    components = [{"name": "chain_0", "kind": "solid", "sprite": current_sprite.copy()}]
    for level in range(1, depth + 1):
        shell_sprite, filled_mask = _build_shell_around(current_mask, colors[level], reserved_sigs)
        components.append({"name": f"chain_{level}", "kind": "shell", "sprite": shell_sprite.copy()})
        next_sprite = shell_sprite.copy()
        hole = _enclosed_holes(shell_sprite != BG)[0]
        hole_r0, hole_c0, _hole_r1, _hole_c1 = hole["bbox_local"]
        paste(next_sprite, current_sprite, hole_r0, hole_c0, transparent=BG)
        current_sprite = next_sprite
        current_mask = filled_mask
        reserved_sigs.add(_mask_signature(filled_mask))
    return components, current_sprite


def _place_components(components: list[dict]) -> tuple[np.ndarray, dict]:
    ordered = sorted(components, key=lambda item: item["sprite"].shape[0] * item["sprite"].shape[1], reverse=True)
    min_h = max(sprite["sprite"].shape[0] for sprite in components) + 1
    min_w = max(sprite["sprite"].shape[1] for sprite in components) + 1
    bbox_area = sum(item["sprite"].shape[0] * item["sprite"].shape[1] for item in components)
    needed_side = int(np.ceil(np.sqrt(bbox_area * 1.45)))

    for _ in range(400):
        lo_h = min(30, max(needed_side, min_h, 24))
        lo_w = min(30, max(needed_side, min_w, 24))
        grid_h = random.randint(lo_h, 30)
        grid_w = random.randint(lo_w, 30)
        grid = np.zeros((grid_h, grid_w), dtype=int)
        occupied = np.zeros((grid_h, grid_w), dtype=int)
        placements: dict[str, tuple[int, int, int, int]] = {}
        for item in ordered:
            sprite = item["sprite"]
            marker = np.ones(sprite.shape, dtype=int)
            try:
                r0, c0 = random_free_location(occupied, marker, transparent=0, border=0, padding=1, padding_connectivity=8)
            except ValueError:
                break
            paste(occupied, marker, r0, c0)
            paste(grid, sprite, r0, c0, transparent=BG)
            placements[item["name"]] = (r0, c0, r0 + sprite.shape[0] - 1, c0 + sprite.shape[1] - 1)
        else:
            return grid, placements
    raise RuntimeError("failed to place components")


def _render_expected(input_grid: np.ndarray, placements: dict[str, tuple[int, int, int, int]], components: list[dict], final_sprite: np.ndarray) -> np.ndarray:
    out = np.zeros_like(input_grid)
    for item in components:
        if item["name"].startswith("chain_") and item["name"] != f"chain_{len([c for c in components if c['name'].startswith('chain_')]) - 1}":
            continue
        box = placements[item["name"]]
        r0, c0, _r1, _c1 = box
        sprite = final_sprite if item["name"] == f"chain_{len([c for c in components if c['name'].startswith('chain_')]) - 1}" else item["sprite"]
        paste(out, sprite, r0, c0, transparent=BG)
    return out


def _has_distance_decoy(placements: dict[str, tuple[int, int, int, int]], shell_names: list[str], depth: int) -> bool:
    for level in range(1, depth + 1):
        child_name = f"chain_{level - 1}"
        parent_name = f"chain_{level}"
        true_dist = _center_distance(placements[child_name], placements[parent_name])
        for shell_name in shell_names:
            if shell_name == parent_name:
                continue
            if _center_distance(placements[child_name], placements[shell_name]) < true_dist:
                return True
    return False


def _validate_sample(input_grid: np.ndarray, expected_output: np.ndarray) -> None:
    solved_output, info = _solve(input_grid)
    assert np.array_equal(solved_output, expected_output), "solver output does not match independently rendered output"
    roots = info["roots"]
    moved_depths = [int(root["depth"]) for root in roots]
    assert moved_depths and max(moved_depths) in {2, 3}, "generator must produce a two-step or three-step chain"
    assert max(moved_depths) >= 2, "recursive assembly is required"
    assert len(info["links"]) >= 2, "at least two assembly steps are required"
    assert any(root["depth"] == 0 for root in roots) or info["remaining_shells"], "sample should keep at least one untouched distractor"
    assert not np.array_equal(input_grid, expected_output), "transformation must be non-trivial"


def _generate_instance() -> tuple[np.ndarray, np.ndarray]:
    depth = random.choice([2, 3])
    color_count = depth + 3
    colors = random.sample(PALETTE, color_count)

    reserved_sigs: set[tuple[tuple[int, int], bytes]] = set()
    chain_components, final_sprite = _build_chain(depth, colors[: depth + 1], reserved_sigs)
    distractor_solid = {"name": "solid_decoy", "kind": "solid", "sprite": _make_unique_solid_sprite(colors[depth + 1], reserved_sigs)}
    reserved_sigs.add(_mask_signature(distractor_solid["sprite"] != BG))
    distractor_shell = {"name": "shell_decoy", "kind": "shell", "sprite": _make_unique_shell(colors[depth + 2], reserved_sigs)}
    components = chain_components + [distractor_solid, distractor_shell]

    for _ in range(200):
        input_grid, placements = _place_components(components)
        if not _has_distance_decoy(placements, ["shell_decoy"] + [f"chain_{level}" for level in range(1, depth + 1)], depth):
            continue
        expected_output = _render_expected(input_grid, placements, components, final_sprite)
        _validate_sample(input_grid, expected_output)
        return input_grid, expected_output
    raise RuntimeError("failed to generate a non-degenerate instance")


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            input_grid, _expected_output = _generate_instance()
        except RuntimeError:
            continue
        if isinstance(input_grid, np.ndarray) and input_grid.ndim == 2 and max(input_grid.shape) <= 30:
            return input_grid
    raise RuntimeError("failed to generate a valid ARC input")
