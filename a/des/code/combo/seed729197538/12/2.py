from __future__ import annotations

import numpy as np


BG = 0
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in DIR8:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        pieces.append(np.asarray(coords, dtype=int))
    return pieces


def _segments(line: np.ndarray) -> list[tuple[int, int, int]]:
    arr = [int(v) for v in line.tolist()]
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < len(arr):
        color = arr[i]
        if color == BG:
            i += 1
            continue
        j = i
        while j + 1 < len(arr) and arr[j + 1] == color:
            j += 1
        runs.append((color, i, j))
        i = j + 1
    return runs


def _detect_channels(grid: np.ndarray) -> tuple[str, dict[int, tuple[int, int]]]:
    top_runs = _segments(grid[0, 1:-1])
    left_runs = _segments(grid[1:-1, 0])
    has_top = bool(top_runs)
    has_left = bool(left_runs)
    assert has_top != has_left, "expected exactly one active border side"

    orientation = "top" if has_top else "left"
    runs = top_runs if has_top else left_runs

    channels: dict[int, tuple[int, int]] = {}
    for color, start, end in runs:
        assert color not in channels, "each channel color must be unique"
        channels[int(color)] = (1 + int(start), 1 + int(end))
    return orientation, channels


def _extract_pieces(grid: np.ndarray) -> dict[int, list[dict[str, object]]]:
    interior = grid[1:-1, 1:-1]
    pieces_by_color: dict[int, list[dict[str, object]]] = {}
    for color in map(int, np.unique(interior)):
        if color == BG:
            continue
        groups = []
        for coords in _components(interior == color):
            rs = coords[:, 0]
            cs = coords[:, 1]
            r0 = int(rs.min())
            c0 = int(cs.min())
            r1 = int(rs.max())
            c1 = int(cs.max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[rs - r0, cs - c0] = True
            groups.append(
                {
                    "mask": mask,
                    "area": int(mask.sum()),
                    "bbox": (r0, c0, r1, c1),
                }
            )
        pieces_by_color[color] = groups
    return pieces_by_color


def _rearrange(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    orientation, channels = _detect_channels(g)
    pieces_by_color = _extract_pieces(g)

    out = g.copy()
    out[1:-1, 1:-1] = BG

    for color, pieces in pieces_by_color.items():
        assert color in channels, "every fragment color must appear on the active border"
        pieces_sorted = sorted(
            pieces,
            key=lambda item: (
                int(item["area"]),
                int(item["bbox"][0]),
                int(item["bbox"][1]),
            ),
        )
        start, end = channels[color]

        if orientation == "left":
            channel_height = end - start + 1
            cursor = 1
            for piece in pieces_sorted:
                mask = piece["mask"]
                ph, pw = mask.shape
                assert ph <= channel_height
                rr = start + (channel_height - ph) // 2
                cc = cursor
                assert cc + pw <= w - 1
                target = out[rr : rr + ph, cc : cc + pw]
                assert np.all(target[mask] == BG)
                target[mask] = color
                cursor += pw + 1
        else:
            channel_width = end - start + 1
            cursor = 1
            for piece in pieces_sorted:
                mask = piece["mask"]
                ph, pw = mask.shape
                assert pw <= channel_width
                rr = cursor
                cc = start + (channel_width - pw) // 2
                assert rr + ph <= h - 1
                target = out[rr : rr + ph, cc : cc + pw]
                assert np.all(target[mask] == BG)
                target[mask] = color
                cursor += ph + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack same-color fragments into row or column channels implied by the border bands.
    """
    return _rearrange(grid)


def _random_connected_mask(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        assert frontier
        choice = tuple(frontier)[int(rng.integers(len(frontier)))]
        cells.add(choice)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _sample_piece(
    rng: np.random.Generator,
    orientation: str,
    channel_span: int,
    used_areas: set[int],
) -> np.ndarray | None:
    for _ in range(200):
        if orientation == "left":
            h = int(rng.integers(1, min(channel_span, 4) + 1))
            w = int(rng.integers(1, 5))
        else:
            h = int(rng.integers(1, 5))
            w = int(rng.integers(1, min(channel_span, 4) + 1))
        area = int(rng.integers(1, h * w + 1))
        mask = _random_connected_mask(rng, h, w, area)
        ph, pw = mask.shape
        if orientation == "left" and ph > channel_span:
            continue
        if orientation == "top" and pw > channel_span:
            continue
        actual_area = int(mask.sum())
        if actual_area in used_areas:
            continue
        return mask
    return None


def _place_scattered_piece(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserve: np.ndarray,
    color: int,
    mask: np.ndarray,
) -> bool:
    h, w = grid.shape
    ph, pw = mask.shape
    for _ in range(200):
        r0 = int(rng.integers(1, h - ph))
        c0 = int(rng.integers(1, w - pw))
        rr0 = max(0, r0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c0 + pw + 1)
        if reserve[rr0:rr1, cc0:cc1].any():
            continue
        patch = grid[r0 : r0 + ph, c0 : c0 + pw]
        patch[mask] = color
        reserve[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _build_sample(rng: np.random.Generator) -> np.ndarray | None:
    orientation = "left" if int(rng.integers(2)) == 0 else "top"
    n_channels = int(rng.integers(2, 5))
    channel_spans = [int(rng.integers(2, 5)) for _ in range(n_channels)]
    colors = [int(v) for v in rng.choice(PALETTE, size=n_channels, replace=False)]

    piece_counts = [int(rng.integers(1, 4)) for _ in range(n_channels)]
    if max(piece_counts) == 1:
        piece_counts[int(rng.integers(n_channels))] = 2

    pieces_by_color: dict[int, list[np.ndarray]] = {}
    max_required_along = 0
    total_pieces = 0

    for color, span, count in zip(colors, channel_spans, piece_counts):
        used_areas: set[int] = set()
        pieces: list[np.ndarray] = []
        for _ in range(count):
            piece = _sample_piece(rng, orientation, span, used_areas)
            if piece is None:
                return None
            pieces.append(piece)
            used_areas.add(int(piece.sum()))
        pieces_by_color[color] = pieces
        along_sizes = [piece.shape[1] if orientation == "left" else piece.shape[0] for piece in pieces]
        max_required_along = max(max_required_along, sum(along_sizes) + len(along_sizes) - 1)
        total_pieces += len(pieces)

    fixed_dim = sum(channel_spans)
    free_dim = max(max_required_along + int(rng.integers(2, 6)), 7 + total_pieces, 10)

    if orientation == "left":
        h, w = fixed_dim + 2, free_dim + 2
    else:
        h, w = free_dim + 2, fixed_dim + 2

    grid = np.zeros((h, w), dtype=int)

    offset = 1
    for color, span in zip(colors, channel_spans):
        if orientation == "left":
            grid[offset : offset + span, 0] = color
        else:
            grid[0, offset : offset + span] = color
        offset += span

    reserve = np.zeros((h, w), dtype=bool)

    to_place: list[tuple[int, np.ndarray]] = []
    for color, pieces in pieces_by_color.items():
        for piece in pieces:
            to_place.append((color, piece))
    rng.shuffle(to_place)
    to_place.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)

    for color, piece in to_place:
        if not _place_scattered_piece(rng, grid, reserve, color, piece):
            return None

    out = _rearrange(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_sample(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
