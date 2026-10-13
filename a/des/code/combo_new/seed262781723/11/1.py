import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPENING_OFFSETS = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
BORDER_OFFSETS = (
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
)

GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
EXTRA_UNUSED_MAX = 2
PLACEMENT_RETRIES = 160
MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _make_envelope(top: int, left: int, direction: str, border_color: int, letter_color: int) -> dict:
    center = (top + 1, left + 1)
    opening = OPENING_OFFSETS[direction]
    border_coords = [
        (top + dr, left + dc)
        for dr, dc in BORDER_OFFSETS
        if (dr, dc) != opening
    ]
    bbox_cells = [(r, c) for r in range(top, top + 3) for c in range(left, left + 3)]
    return {
        "top": top,
        "left": left,
        "center": center,
        "direction": direction,
        "vector": DIR_VECTORS[direction],
        "border_color": border_color,
        "letter_color": letter_color,
        "border_coords": border_coords,
        "bbox_cells": bbox_cells,
    }


def _match_envelope(window: np.ndarray) -> tuple[str, int, int] | None:
    if window.shape != (3, 3):
        return None
    center = int(window[1, 1])
    if center == BACKGROUND:
        return None

    for direction, opening in OPENING_OFFSETS.items():
        border_color = None
        ok = True
        for dr, dc in BORDER_OFFSETS:
            value = int(window[dr, dc])
            if (dr, dc) == opening:
                if value != BACKGROUND:
                    ok = False
                    break
                continue
            if value == BACKGROUND:
                ok = False
                break
            if border_color is None:
                border_color = value
            elif value != border_color:
                ok = False
                break
        if ok and border_color is not None and border_color != center:
            return direction, border_color, center
    return None


def _find_envelopes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    envelopes: list[dict] = []
    used_cells: set[tuple[int, int]] = set()

    for top in range(h - 2):
        for left in range(w - 2):
            match = _match_envelope(g[top : top + 3, left : left + 3])
            if match is None:
                continue

            direction, border_color, letter_color = match
            env = _make_envelope(top, left, direction, border_color, letter_color)
            payload = set(env["border_coords"]) | {env["center"]}
            if payload & used_cells:
                continue
            used_cells |= payload
            envelopes.append(env)

    assert envelopes, "expected at least one envelope"
    return envelopes


def _find_mailboxes(grid: np.ndarray, envelopes: list[dict]) -> dict[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    envelope_cells = {
        cell
        for env in envelopes
        for cell in env["border_coords"] + [env["center"]]
    }
    mailboxes: dict[tuple[int, int], int] = {}

    for r, c in np.argwhere(g != BACKGROUND):
        pos = (int(r), int(c))
        if pos in envelope_cells:
            continue

        isolated = True
        for rr in range(max(0, pos[0] - 1), min(h, pos[0] + 2)):
            for cc in range(max(0, pos[1] - 1), min(w, pos[1] + 2)):
                if (rr, cc) != pos and int(g[rr, cc]) != BACKGROUND:
                    isolated = False
                    break
            if not isolated:
                break
        if isolated:
            mailboxes[pos] = int(g[pos])

    return mailboxes


def _ray_cells(envelope: dict, shape: tuple[int, int], start_distance: int) -> list[tuple[int, int]]:
    r, c = envelope["center"]
    dr, dc = envelope["vector"]
    cells: list[tuple[int, int]] = []
    step = start_distance
    while True:
        rr = r + step * dr
        cc = c + step * dc
        if not _in_bounds(shape, rr, cc):
            break
        cells.append((rr, cc))
        step += 1
    return cells


def _first_mailbox(envelope: dict, mailboxes: dict[tuple[int, int], int], shape: tuple[int, int]) -> tuple[int, int] | None:
    for pos in _ray_cells(envelope, shape, start_distance=1):
        if pos in mailboxes:
            return pos
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Deliver each envelope's letter color to the first mailbox along its opening direction and return the mailbox color to the frame.

    Input structure:
    - The background is black.
    - Some objects are 3x3 envelopes whose seven visible border pixels share one color, whose center pixel has a different color, and whose missing side-middle border pixel marks the opening direction.
    - Other foreground objects are isolated single-pixel mailboxes that do not touch any envelope.

    Rule:
    - Detect every envelope and read its opening direction from the missing border pixel.
    - Cast a straight ray from the envelope center through that opening and find the first mailbox on the ray, if any.
    - Recolor that mailbox to the envelope center's letter color.
    - Recolor all visible border pixels of the envelope to the mailbox's original color, while leaving the center letter unchanged.
    - Envelopes with no mailbox on their ray and mailboxes hit by no ray stay unchanged.

    Key insight:
    - The gap in each 3x3 frame encodes both which ray to follow and which mailbox color must flow back to recolor the envelope border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    envelopes = _find_envelopes(g)
    mailboxes = _find_mailboxes(g, envelopes)
    out = g.copy()

    for env in envelopes:
        target = _first_mailbox(env, mailboxes, g.shape)
        if target is None:
            continue

        original_mailbox_color = mailboxes[target]
        out[target] = int(env["letter_color"])
        for r, c in env["border_coords"]:
            out[r, c] = original_mailbox_color

    return out


def _mark_padded(mask: np.ndarray, top: int, left: int) -> None:
    h, w = mask.shape
    r0 = max(0, top - 1)
    r1 = min(h, top + 4)
    c0 = max(0, left - 1)
    c1 = min(w, left + 4)
    mask[r0:r1, c0:c1] = True


def _can_place_envelope(mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    if top < 0 or left < 0 or top + 3 > h or left + 3 > w:
        return False
    r0 = max(0, top - 1)
    r1 = min(h, top + 4)
    c0 = max(0, left - 1)
    c1 = min(w, left + 4)
    return not bool(mask[r0:r1, c0:c1].any())


def _mailbox_far_from_others(pos: tuple[int, int], mailboxes: dict[tuple[int, int], int]) -> bool:
    r, c = pos
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) in mailboxes:
                return False
    return True


def _sample_target_flags(count: int) -> list[bool]:
    assert count >= 2
    flags = [bool(random.getrandbits(1)) for _ in range(count)]
    if all(flags):
        flags[random.randrange(count)] = False
    if not any(flags):
        flags[random.randrange(count)] = True
    return flags


def _place_envelopes(h: int, w: int, count: int) -> list[dict] | None:
    envelopes: list[dict] = []
    padded_mask = np.zeros((h, w), dtype=bool)
    directions = tuple(DIR_VECTORS)

    for _ in range(count):
        placed = False
        for _attempt in range(PLACEMENT_RETRIES):
            top = random.randint(0, h - 3)
            left = random.randint(0, w - 3)
            if not _can_place_envelope(padded_mask, top, left):
                continue
            border_color, letter_color = random.sample(PALETTE, 2)
            direction = random.choice(directions)
            env = _make_envelope(top, left, direction, border_color, letter_color)
            envelopes.append(env)
            _mark_padded(padded_mask, top, left)
            placed = True
            break
        if not placed:
            return None

    return envelopes


def _envelope_corridors_clear(envelopes: list[dict], shape: tuple[int, int]) -> bool:
    bbox_sets = [set(env["bbox_cells"]) for env in envelopes]
    for i, env in enumerate(envelopes):
        ray = set(_ray_cells(env, shape, start_distance=2))
        for j, bbox in enumerate(bbox_sets):
            if i != j and ray & bbox:
                return False
    return True


def _mailbox_candidates(
    envelope_index: int,
    envelopes: list[dict],
    corridors: list[list[tuple[int, int]]],
    envelope_padding: np.ndarray,
    mailboxes: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    other_corridor_cells = {
        cell
        for j, corridor in enumerate(corridors)
        if j != envelope_index
        for cell in corridor
    }
    candidates: list[tuple[int, int]] = []
    for pos in corridors[envelope_index]:
        if pos in other_corridor_cells:
            continue
        if envelope_padding[pos]:
            continue
        if pos in mailboxes:
            continue
        if not _mailbox_far_from_others(pos, mailboxes):
            continue
        candidates.append(pos)
    return candidates


def _paint_grid(shape: tuple[int, int], envelopes: list[dict], mailboxes: dict[tuple[int, int], int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for env in envelopes:
        for r, c in env["border_coords"]:
            grid[r, c] = int(env["border_color"])
        cr, cc = env["center"]
        grid[cr, cc] = int(env["letter_color"])
    for (r, c), color in mailboxes.items():
        grid[r, c] = int(color)
    return grid


def _validate_generated(
    grid: np.ndarray,
    envelopes: list[dict],
    target_flags: list[bool],
    target_map: dict[tuple[int, int], tuple[int, int]],
    mailboxes: dict[tuple[int, int], int],
) -> bool:
    detected = _find_envelopes(grid)
    if len(detected) != len(envelopes):
        return False

    detected_by_center = {env["center"]: env for env in detected}
    if set(detected_by_center) != {env["center"] for env in envelopes}:
        return False

    detected_mailboxes = _find_mailboxes(grid, detected)
    if detected_mailboxes != mailboxes:
        return False

    for env, should_hit in zip(envelopes, target_flags):
        center = env["center"]
        hit = _first_mailbox(detected_by_center[center], detected_mailboxes, grid.shape)
        if should_hit:
            if hit != target_map.get(center):
                return False
        elif hit is not None:
            return False

    return not np.array_equal(main(grid), grid)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    area = h * w
    max_envelopes = 3 if area < 220 else 4 if area < 360 else 5
    count = random.randint(2, max_envelopes)
    target_flags = _sample_target_flags(count)

    envelopes = _place_envelopes(h, w, count)
    if envelopes is None:
        return None
    if not _envelope_corridors_clear(envelopes, (h, w)):
        return None

    envelope_padding = np.zeros((h, w), dtype=bool)
    for env in envelopes:
        _mark_padded(envelope_padding, env["top"], env["left"])

    corridors = [_ray_cells(env, (h, w), start_distance=3) for env in envelopes]
    if any(target_flags[i] and not corridors[i] for i in range(count)):
        return None

    mailboxes: dict[tuple[int, int], int] = {}
    target_map: dict[tuple[int, int], tuple[int, int]] = {}

    for i, env in enumerate(envelopes):
        if not target_flags[i]:
            continue
        candidates = _mailbox_candidates(i, envelopes, corridors, envelope_padding, mailboxes)
        if not candidates:
            return None
        target = random.choice(candidates)
        color_choices = [color for color in PALETTE if color not in {env["border_color"], env["letter_color"]}]
        mailboxes[target] = random.choice(color_choices)
        target_map[env["center"]] = target

    extra_unused = 0 if random.random() < 0.5 else random.randint(1, EXTRA_UNUSED_MAX)
    all_corridor_cells = {cell for corridor in corridors for cell in corridor}
    for _ in range(extra_unused):
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if not envelope_padding[r, c]
            and (r, c) not in all_corridor_cells
            and (r, c) not in mailboxes
            and _mailbox_far_from_others((r, c), mailboxes)
        ]
        if not candidates:
            return None
        pos = random.choice(candidates)
        mailboxes[pos] = random.choice(PALETTE)

    grid = _paint_grid((h, w), envelopes, mailboxes)
    if not _validate_generated(grid, envelopes, target_flags, target_map, mailboxes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
