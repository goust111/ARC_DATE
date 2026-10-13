import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, get_bg, paste, random_free_location, random_sprite


BG = int(Color.BLACK)
ORIENTATIONS = ("up", "right", "down", "left")
DIRECTIONS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
ROTATE_K = {
    "up": 0,
    "right": -1,
    "down": 2,
    "left": 1,
}
ARROW_MASKS = {
    "up": np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    "right": np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    "down": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    "left": np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
}
ARROW_TIPS = {
    "up": (0, 1),
    "right": (1, 1),
    "down": (1, 1),
    "left": (1, 0),
}
VALID_COUNT_TEMPLATES = [
    (2, 1, 0, 0),
    (3, 1, 1, 0),
    (3, 2, 1, 0),
    (4, 1, 1, 1),
    (4, 2, 1, 1),
]
SPRITE_RETRIES = 200
GENERATION_RETRIES = 400
SPRITE_BORDER = 5


def _arrow_patch(orientation: str, color: int) -> np.ndarray:
    patch = np.full(ARROW_MASKS[orientation].shape, BG, dtype=int)
    patch[ARROW_MASKS[orientation]] = int(color)
    return patch


def _majority_orientation(counts: dict[str, int] | Counter) -> str | None:
    tallies = {orientation: int(counts.get(orientation, 0)) for orientation in ORIENTATIONS}
    best = max(tallies.values(), default=0)
    if best <= 0:
        return None
    winners = [orientation for orientation, value in tallies.items() if value == best]
    if len(winners) != 1:
        return None
    return winners[0]


def _ray_trace(
    sprite_mask: np.ndarray,
    tip: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    height, width = map(int, shape)
    dr, dc = map(int, direction)
    r, c = int(tip[0]) + dr, int(tip[1]) + dc
    path: list[tuple[int, int]] = []
    while 0 <= r < height and 0 <= c < width:
        if bool(sprite_mask[r, c]):
            return path, True
        path.append((int(r), int(c)))
        r += dr
        c += dc
    return path, False


def _classify_arrow(coords: np.ndarray) -> tuple[str, tuple[int, int]]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    local = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[local[:, 0], local[:, 1]] = True
    for orientation, template in ARROW_MASKS.items():
        if mask.shape == template.shape and np.array_equal(mask, template):
            tip_r, tip_c = ARROW_TIPS[orientation]
            return orientation, (int(r0 + tip_r), int(c0 + tip_c))
    raise AssertionError("unexpected arrow component")


def _world_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    locals_ = np.argwhere(mask)
    return {(int(r0 + r), int(c0 + c)) for r, c in locals_}


def _can_place_arrow(
    grid: np.ndarray,
    patch_mask: np.ndarray,
    r0: int,
    c0: int,
    ray_forbidden: set[tuple[int, int]],
) -> bool:
    height, width = map(int, grid.shape)
    patch_h, patch_w = map(int, patch_mask.shape)
    if r0 < 0 or c0 < 0 or r0 + patch_h > height or c0 + patch_w > width:
        return False

    cells = _world_cells(patch_mask, r0, c0)
    for r, c in cells:
        if int(grid[r, c]) != BG or (r, c) in ray_forbidden:
            return False

    for r, c in cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells and int(grid[nr, nc]) != BG:
                return False
    return True


def _row_spans(sprite_mask: np.ndarray) -> dict[int, tuple[int, int]]:
    spans: dict[int, tuple[int, int]] = {}
    for r in np.where(sprite_mask.any(axis=1))[0]:
        cols = np.where(sprite_mask[int(r)])[0]
        spans[int(r)] = (int(cols.min()), int(cols.max()))
    return spans


def _col_spans(sprite_mask: np.ndarray) -> dict[int, tuple[int, int]]:
    spans: dict[int, tuple[int, int]] = {}
    for c in np.where(sprite_mask.any(axis=0))[0]:
        rows = np.where(sprite_mask[:, int(c)])[0]
        spans[int(c)] = (int(rows.min()), int(rows.max()))
    return spans


def _tip_to_topleft(orientation: str, tip: tuple[int, int]) -> tuple[int, int]:
    tip_r, tip_c = map(int, tip)
    local_r, local_c = ARROW_TIPS[orientation]
    return int(tip_r - local_r), int(tip_c - local_c)


def _candidate_tips(sprite_mask: np.ndarray, orientation: str, hits_sprite: bool) -> list[tuple[int, int]]:
    height, width = map(int, sprite_mask.shape)
    candidates: list[tuple[int, int]] = []
    if orientation == "right":
        for r, (min_c, max_c) in _row_spans(sprite_mask).items():
            tip_range = range(1, min_c - 1) if hits_sprite else range(max_c + 2, width)
            candidates.extend((int(r), int(tip_c)) for tip_c in tip_range)
    elif orientation == "left":
        for r, (min_c, max_c) in _row_spans(sprite_mask).items():
            tip_range = range(max_c + 2, width - 1) if hits_sprite else range(0, min_c - 1)
            candidates.extend((int(r), int(tip_c)) for tip_c in tip_range)
    elif orientation == "up":
        for c, (min_r, max_r) in _col_spans(sprite_mask).items():
            tip_range = range(max_r + 2, height - 1) if hits_sprite else range(0, min_r - 1)
            candidates.extend((int(tip_r), int(c)) for tip_r in tip_range)
    else:
        for c, (min_r, max_r) in _col_spans(sprite_mask).items():
            tip_range = range(1, min_r - 1) if hits_sprite else range(max_r + 2, height)
            candidates.extend((int(tip_r), int(c)) for tip_r in tip_range)
    random.shuffle(candidates)
    return candidates


def _candidate_placements(
    grid: np.ndarray,
    sprite_mask: np.ndarray,
    orientation: str,
    hits_sprite: bool,
    ray_forbidden: set[tuple[int, int]],
) -> list[tuple[int, int, tuple[int, int], list[tuple[int, int]]]]:
    candidates: list[tuple[int, int, tuple[int, int], list[tuple[int, int]]]] = []
    patch_mask = ARROW_MASKS[orientation]
    for tip in _candidate_tips(sprite_mask, orientation, hits_sprite):
        r0, c0 = _tip_to_topleft(orientation, tip)
        if not _can_place_arrow(grid, patch_mask, r0, c0, ray_forbidden):
            continue
        path, actually_hits = _ray_trace(sprite_mask, tip, DIRECTIONS[orientation], grid.shape)
        if bool(actually_hits) != bool(hits_sprite):
            continue
        if any(int(grid[r, c]) != BG for r, c in path):
            continue
        candidates.append((int(r0), int(c0), (int(tip[0]), int(tip[1])), path))
    random.shuffle(candidates)
    return candidates


def _build_sprite() -> np.ndarray:
    for _ in range(SPRITE_RETRIES):
        height = random.randint(4, 7)
        width = random.randint(4, 7)
        palette_size = random.randint(2, 4)
        palette = random.sample(Color.NOT_BLACK, k=palette_size)
        sprite = random_sprite(
            height,
            width,
            symmetry="not_symmetric",
            color_palette=palette,
            connectivity=4,
            background=BG,
        )
        sprite_box = bbox(sprite != BG)
        if sprite_box is None:
            continue
        sprite = crop_bbox(sprite, sprite_box)
        if sprite.shape[0] < 4 or sprite.shape[1] < 4:
            continue
        if int(np.count_nonzero(sprite != BG)) < 8:
            continue
        if len(np.unique(sprite[sprite != BG])) < 2:
            continue
        if any(np.array_equal(sprite, np.rot90(sprite, k)) for k in (1, 2, 3)):
            continue
        return sprite
    raise RuntimeError("failed to generate a non-symmetric sprite")


def _sum_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {orientation: int(a.get(orientation, 0)) + int(b.get(orientation, 0)) for orientation in ORIENTATIONS}


def _sample_valid_counts(winner: str) -> dict[str, int]:
    counts = {orientation: 0 for orientation in ORIENTATIONS}
    template = random.choice(VALID_COUNT_TEMPLATES)
    counts[winner] = int(template[0])
    others = [orientation for orientation in ORIENTATIONS if orientation != winner]
    random.shuffle(others)
    for orientation, value in zip(others, template[1:]):
        counts[orientation] = int(value)
    return counts


def _sample_invalid_counts(valid_counts: dict[str, int], winner: str, wants_mismatch: bool) -> dict[str, int]:
    others = [orientation for orientation in ORIENTATIONS if orientation != winner]
    for _ in range(100):
        invalid = {orientation: 0 for orientation in ORIENTATIONS}
        if wants_mismatch:
            decoy = random.choice(others)
            invalid[decoy] = int(valid_counts[winner] - valid_counts[decoy] + random.randint(1, 2))
            extra_budget = random.randint(0, 2)
            for _ in range(extra_budget):
                invalid[random.choice(ORIENTATIONS)] += 1
        else:
            extra_budget = random.randint(1, 4)
            for _ in range(extra_budget):
                invalid[random.choice(ORIENTATIONS)] += 1
        total_counts = _sum_counts(valid_counts, invalid)
        total_winner = _majority_orientation(total_counts)
        if total_winner is None:
            continue
        if wants_mismatch and total_winner != winner:
            return invalid
        if not wants_mismatch and total_winner == winner:
            return invalid
    raise RuntimeError("failed to sample invalid vote counts")


def _place_arrows(
    grid: np.ndarray,
    sprite_mask: np.ndarray,
    valid_counts: dict[str, int],
    invalid_counts: dict[str, int],
    arrow_palette: list[int],
) -> list[dict] | None:
    placements: list[dict] = []
    ray_forbidden: set[tuple[int, int]] = set()
    jobs: list[tuple[str, bool]] = []
    for orientation in ORIENTATIONS:
        jobs.extend((orientation, True) for _ in range(valid_counts[orientation]))
        jobs.extend((orientation, False) for _ in range(invalid_counts[orientation]))
    random.shuffle(jobs)

    for orientation, hits_sprite in jobs:
        candidates = _candidate_placements(grid, sprite_mask, orientation, hits_sprite, ray_forbidden)
        if not candidates:
            return None
        r0, c0, tip, path = random.choice(candidates)
        color = int(random.choice(arrow_palette))
        paste(grid, _arrow_patch(orientation, color), r0, c0, transparent=BG)
        placements.append(
            {
                "orientation": orientation,
                "valid": bool(hits_sprite),
                "tip": tip,
                "path": path,
            }
        )
        ray_forbidden.update(path)
    return placements


def _validate_instance(
    grid: np.ndarray,
    sprite: np.ndarray,
    sprite_mask: np.ndarray,
    placements: list[dict],
    winner: str,
    wants_mismatch: bool,
) -> bool:
    valid_counts = Counter()
    total_counts = Counter()

    for placement in placements:
        orientation = placement["orientation"]
        total_counts[orientation] += 1
        path, hits = _ray_trace(sprite_mask, placement["tip"], DIRECTIONS[orientation], grid.shape)
        if bool(hits) != bool(placement["valid"]):
            return False
        if path != placement["path"]:
            return False
        if any(int(grid[r, c]) != BG for r, c in path):
            return False
        if hits:
            valid_counts[orientation] += 1

    if _majority_orientation(valid_counts) != winner:
        return False
    total_winner = _majority_orientation(total_counts)
    if total_winner is None:
        return False
    if wants_mismatch and total_winner == winner:
        return False
    if not wants_mismatch and total_winner != winner:
        return False

    expected = np.rot90(sprite, ROTATE_K[winner])
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the arrows whose sightline hits the large sprite, then rotate the cropped sprite by the winning vote direction.

    Input structure:
    - The grid has one multicolor large sprite on a background plus several disconnected four-cell arrow markers.
    - Each arrow is one of the four axis-aligned rotations of the same tiny tetromino shape.
    - Following an arrow's pointing direction from its tip either reaches a large-sprite foreground cell or exits the grid.

    Rule:
    - Detect the unique non-arrow connected component as the large sprite and crop it to its tight bounding box.
    - For every four-cell arrow, cast a straight ray from its tip in its facing direction; count only the arrows whose ray reaches the large sprite before leaving the grid.
    - Find the unique majority direction among those valid votes and rotate the cropped sprite by up=0 degrees, right=clockwise 90 degrees, down=180 degrees, left=counterclockwise 90 degrees.

    Key insight:
    - Arrow orientation alone is not enough: only arrows with geometric line-of-sight to the large sprite are eligible to vote, and that vote chooses a whole-sprite rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    sprite_candidates = [component for component in components if int(component["area"]) > 4]
    assert len(sprite_candidates) == 1, "expected exactly one large sprite"
    sprite_component = sprite_candidates[0]

    arrow_components = [component for component in components if component is not sprite_component]
    assert arrow_components, "expected at least one arrow"
    assert all(int(component["area"]) == 4 for component in arrow_components), "unexpected non-arrow component"

    sprite_mask = np.zeros(g.shape, dtype=bool)
    sprite_coords = sprite_component["coords"]
    sprite_mask[sprite_coords[:, 0], sprite_coords[:, 1]] = True
    sprite = crop_bbox(g, sprite_component["bbox"])

    valid_votes: Counter[str] = Counter()
    for component in arrow_components:
        orientation, tip = _classify_arrow(component["coords"])
        _path, hits = _ray_trace(sprite_mask, tip, DIRECTIONS[orientation], g.shape)
        if hits:
            valid_votes[orientation] += 1

    winner = _majority_orientation(valid_votes)
    assert winner is not None, "expected a unique valid-vote majority"
    return np.rot90(sprite, ROTATE_K[winner])


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        sprite = _build_sprite()
        sprite_colors = set(map(int, np.unique(sprite[sprite != BG])))
        arrow_colors = [color for color in Color.NOT_BLACK if int(color) not in sprite_colors]
        arrow_palette = random.sample(arrow_colors, k=random.randint(1, min(2, len(arrow_colors))))

        height = random.randint(max(16, sprite.shape[0] + 10), min(30, sprite.shape[0] + 18))
        width = random.randint(max(16, sprite.shape[1] + 10), min(30, sprite.shape[1] + 18))
        grid = np.full((height, width), BG, dtype=int)

        try:
            sprite_r, sprite_c = random_free_location(
                grid,
                sprite,
                transparent=BG,
                border=SPRITE_BORDER,
                padding=0,
                padding_connectivity=4,
            )
        except ValueError:
            continue
        paste(grid, sprite, sprite_r, sprite_c, transparent=BG)

        sprite_mask = np.zeros_like(grid, dtype=bool)
        sprite_mask[sprite_r : sprite_r + sprite.shape[0], sprite_c : sprite_c + sprite.shape[1]] = sprite != BG

        winner = random.choice(ORIENTATIONS)
        valid_counts = _sample_valid_counts(winner)
        wants_mismatch = random.random() < 0.6
        try:
            invalid_counts = _sample_invalid_counts(valid_counts, winner, wants_mismatch)
        except RuntimeError:
            continue

        placements = _place_arrows(grid, sprite_mask, valid_counts, invalid_counts, arrow_palette)
        if placements is None:
            continue

        comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False)
        arrow_areas = sorted(int(component["area"]) for component in comps)
        expected_arrows = sum(valid_counts.values()) + sum(invalid_counts.values())
        if arrow_areas.count(4) != expected_arrows or max(arrow_areas, default=0) <= 4:
            continue

        if _validate_instance(grid, sprite, sprite_mask, placements, winner, wants_mismatch):
            return grid

    raise RuntimeError("failed to generate a valid input")
