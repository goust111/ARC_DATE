你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="58f5dbd5" lines="20-34">
<description>
Carve color-matched motifs into solid blocks and pack them into a compact grid.

Input structure:
- bg = most common color.
- Several solid 5×5 monochrome blocks, each a distinct non-bg color.
- For each block color, a small same-colored motif sprite elsewhere on the grid.

Rule:
- Match each motif to its same-colored 5×5 block.
- Center the motif within the block and recolor those cells to bg (punching holes).
- Pack all carved blocks into an output grid preserving their original row/column order, separated by 1-cell gaps and a 1-cell outer border.

Key insight: The motif acts as a stencil — it's not stamped onto the block but carved out of it as background holes.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="97d7923e" lines="81-94">
<description>
Recolor vertical bars by legend-driven rank selection.

Input structure:
- bg = background. Top-left: monochrome blocks touching row 0; each has color K and height k.
- Main grid: vertical bars capped top and bottom by a single color, with solid-color interiors.

Rule:
- For each legend entry (K, k), find vertical bars capped by K.
- Rank bars by interior length (longest = rank 1).
- Recolor the rank-k bar's interior to K.

Key insight: Legend height is a rank index into length-sorted bars, not a literal size.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="a47bf94d" lines="178-189">
<description>
Transmit colored tokens along wire networks, converting each to its complementary 3×3 pattern.

Input structure:
- bg = background. Non-bg cells form wire networks and 3×3 tokens (solid blocks or X/diagonal patterns) near wire endpoints.

Rule:
- Every solid 3×3 token is replaced in-place by a diamond (center + 4 cardinal neighbors).
- Each token near a wire endpoint sends its complement to the opposite endpoint: solid emits X, X emits diamond.

Key insight: Token shape cycles solid → X → diamond; wires route each complement to the far end.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="e3721c99" lines="69-83">
<description>
Recolor grey shapes by matching their hole count to a color legend.

Input structure:
- bg = most common color; grey (5) sprites scattered across the grid.
- Colored legend shapes (non-bg, non-grey, at least 2×2, not touching the outer border) near grid edge,
  each with a distinct number of enclosed holes.

Rule:
- Each colored legend shape defines a mapping: hole count → color.
- For every grey sprite, count its holes (enclosed bg regions not touching the sprite's bounding-box border).
- Recolor the grey sprite to the matching legend color; erase grey distractors with unlisted hole counts.

Key insight: Shape identity is encoded by topology (number of holes), not geometry or size.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="a6f40cea" lines="74-88">
<description>
Complete partial paths inside a framed rectangle by projecting external color traces inward.

Input structure:
- bg, a 1px rectangular frame (border color) with uniform interior fill.
- Colored components outside the frame touching its edges.

Rule:
- Output = the frame's interior, filled with the fill color.
- Map each outside component's frame-touch points into interior coordinates.
- Monochrome components become solid Manhattan stripes between mapped endpoints.
- Two-color components become alternating-color Manhattan paths.

Key insight: Endpoint colors of two-color traces anchor the alternating phase deterministically.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="21897d95" lines="283-301">
<description>
Repaint regions from T-shaped control markers, erase the markers, and optionally rotate based on a color-9 L-marker.

Input structure:
- The grid is a mosaic of monochrome regions.
- Color 1 forms T-shaped control markers on region boundaries.
- Some controls use three singleton color-1 pixels around a colored center; that center is a literal replacement color.
- An optional 5-cell L-shaped color-9 marker triggers 90-degree rotation of the output.

Rule:
- For a color-1 T marker, the single stem points at the destination region to repaint.
- The two diagonal cells behind the stem identify the source region; the destination gets that source region's original color.
- For a colored-center T marker, the same stem identifies the destination, but the center color is the replacement.
- All marker cells are filled with the final color of the source-side region.
- A 5-cell L-shaped color-9 marker controls rotation: ┐ (corner top-right) → CCW 90°, ┌ (corner top-left) → CW 90°.

Key insight:
- The stem chooses what changes, while the two diagonals behind the stem choose where the marker itself belongs.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="7b0280bc" lines="54-69">
<description>
Recolor the shortest path between two terminal squares in a square-and-track graph.

Input structure:
- bg = most common color.
- Two non-bg colors form solid NxN squares (N>=2): one color marks exactly 2 terminals, the other marks stations.
- A third non-bg color forms 1-cell-wide track segments connecting pairs of squares (graph edges).

Rule:
- Find the shortest path (fewest edges) between the two terminal squares.
- Recolor every station square on that path to green.
- Recolor every track segment along that path to grey.
- Everything else unchanged.

Key insight: The grid encodes a graph where squares are nodes and tracks are edges; the task is graph shortest-path, not spatial proximity.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="4c416de3" lines="99-118">
<description>
Replicate a pre-drawn corner pattern to every other marker-bearing corner.

Input structure:
- bg = most frequent color. Hollow 0-frames (possibly partial L-shapes); each
  frame-corner cell is either 0 or a painted (non-bg, non-0) color.
- >= 1 'source' corner: the pattern is already drawn (one color, both outward
  cells and an inward-diagonal trail).
- Other corners: a single colored pixel on the inward diagonal at the pattern's
  deepest (k,k) offset; the pixel's color is that corner's target color.

Rule:
- Extract the canonical (TL-oriented) pattern from any source.
- For each non-source corner whose marker cell is colored, paint the same
  pattern rotated to that corner's orientation, using the marker's color.

Key insight: the pattern is symmetric about the corner's inward-diagonal axis
(canonical dr == dc), so it reflects cleanly across all four corner orientations.
Painting uses each target's OWN marker color, not the source color.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="eee78d87" lines="40-55">
<description>
Decode the colored 3x3 contour motif into a 16x16 periodic lattice with the motif stamped in maroon at the center.

Input structure:
- The grid is 6x6 with orange background and one non-orange color.
- The non-orange pixels have a tight 3x3 bbox containing one contour glyph.
- The glyph is one of the task's period-3 contour forms: plus, H, X, or four corners.

Rule:
- Decode the 3x3 glyph into the period-3 orange phase tile.
- Fill a black 16x16 canvas by repeating that orange phase tile.
- Tile the original 3x3 glyph 2x2 and recolor its true pixels to maroon in the centered 6x6 region.

Key insight:
- The colored object is not copied at its input location; it is a small contour code whose phase tells where orange appears in the repeated lattice.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="271d71e2" lines="9-22">
<description>
Move bordered containers toward same-colored destination markers, charging gray cells to orange.

Input structure:
- bg = 6 (magenta). Black-bordered (0) rectangles with gray (5) / orange (7) interiors.
- A head sprite (9) sits adjacent to one container edge; a matching destination sprite lies along the movement axis.

Rule:
- Each container translates toward its destination along the head-to-dest axis.
- Steps = min(head-to-dest distance, number of gray interior cells).
- Gray cells become orange from the head-facing side inward, one per step.

Key insight: Gray cell count caps movement — a container stops short if it runs out of gray cells to charge.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="2d0172a1" lines="209-224">
<description>
Convert a rough nested line drawing into a clean canonical frame-and-dot diagram.

Input structure:
- The grid has one background color and one foreground color.
- Foreground 4-connected objects are rough closed frames or small filled blobs.
- Frames may contain other frames/blobs; blobs outside a frame remain outside it.

Rule:
- Detect the containment tree of foreground objects using holes of each frame object.
- Replace every blob with one foreground pixel.
- Replace every frame with a rectangular border padded two cells around its direct children.
- Place siblings by their original row/column overlap and order, with one empty cell between canonical objects.

Key insight: The exact wobbly geometry is irrelevant; only enclosure plus relative sibling positions define the output.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="5dbc8537" lines="10-23">
<description>
Fill rectangular cavities in a template shape by placing source sprites into matching holes.

Input structure:
- bg = most common color. A dominant "base" color forms a large frame/template with rectangular gaps.
- Small colored sprites sit outside the template region.

Rule:
- The base color (largest connected component excluding bg) defines the template bbox.
- Each rectangular hole inside the template matches exactly one outside sprite by size.
- Place each sprite into its matching cavity; output is the filled template bbox.

Key insight: Sprite bounding boxes match cavity dimensions exactly — it's a jigsaw-fit puzzle where each piece has a unique slot.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="bf45cf4b" lines="7-19">
<description>
Scale up a monochrome control sprite by stamping a polychrome tile at each filled pixel.

Input structure:
- bg = background color.
- Two foreground objects (8-connected): a monochrome rectangular sprite with bg holes (control), and a polychrome square tile (block).

Rule:
- For each cell in the control sprite's bbox: if filled, stamp a copy of the block tile; if bg, leave a tile-sized bg region.
- Output grid dimensions are (control_h × tile_h) by (control_w × tile_w).

Key insight: The control sprite acts as a binary mask that gets "pixel-scaled" by the tile — each pixel expands to a full tile copy or empty space.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="136b0064" lines="25-39">
<description>
Draw a polyline by decoding 3x3 glyph instructions read across a divided grid.

Input structure:
- Left of a full-height vertical divider: 3x3 colored glyphs on bg.
- Right of divider: a single gray (5) start pixel.

Rule:
- Remove the divider; shift the gray start into the left canvas.
- Read glyphs in column-major order (left→right, top→bottom).
- Each glyph's shape (pixel area) encodes a segment: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
- Append segments sequentially, stepping down one cell between each.

Key insight: Glyph pixel count (area) uniquely determines both segment length and direction — the shape itself is the instruction.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="7ed72f31" lines="8-22">
<description>
Mirror colored sprites using red (2) markers as reflection axes.

Input structure:
- bg = most common color. Red (2) = marker color.
- Colored sprites appear near red markers.

Rule:
- Red line within a sprite (vertical or horizontal) → mirror payload across that axis.
- Single red pixel in a mixed sprite → point-reflect payload around it.
- Isolated red pixel (alone, no payload) → point-reflect each adjacent 8-connected color component.
- Originals remain; mirrored pixels overlay.

Key insight: Red marker geometry (line vs single pixel) determines reflection type; isolated vs embedded red determines what gets reflected.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="6e453dd6" lines="39-52">
<description>
Slide black shapes rightward to a grey separator, then shoot red rays from enclosed holes.

Input structure:
- A vertical grey (5) separator column divides the grid into left and right halves.
- Black shapes sit on the left side against a background color (bg).

Rule:
- Translate each black connected component rightward until its right edge is flush with the separator.
- After sliding, find enclosed bg regions (holes) within the black shapes that touch the column just left of the separator.
- For each row spanned by such a hole, emit a red ray across the entire right half of the grid.

Key insight: Only holes adjacent to the separator trigger rays — the sliding creates enclosures that act as "nozzles" projecting rightward.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="9aaea919" lines="38-51">
<description>
Transfer sprites from red-marked lanes to green-marked lanes, recoloring and stacking.

Input structure:
- bg = most common color. Red (2) and green (3) thin bars define vertical lanes.
- Each lane contains repeated identical monochrome sprites in an evenly spaced vertical grid.

Rule:
- Each red lane's sprites turn grey (5); their count transfers to the nearest green lane.
- New sprites fill empty lattice slots bottom-up, using the source shape recolored to the destination lane's sprite color.
- All marker bars are removed.

Key insight: Sprite count is conserved across lanes — red sprites become grey, and equally many new sprites stack into the green lane's regular grid.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="b9e38dc0" lines="7-18">
<description>
Fill the region enclosed by a wall sprite and radiate color outward through its opening.

Input structure:
- bg = most common color. Wall = most frequent non-bg color, forming a partial enclosure (e.g. U-shape).
- A fill-color pixel sits inside the enclosed area. Other non-bg pixels may act as obstacles.
Rule:
- Flood-fill the enclosed area with the fill color, radiating outward through the wall's open end.
- Obstacles cast "shadows" along the radiate direction, blocking fill behind them.

Key insight: The wall endpoints define the opening direction; fill radiates outward through it, and obstacles block propagation along that same axis.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="9385bd28" lines="71-87">
<description>
Fill rectangles between diagonal marker pairs using a color legend.

Input structure:
- bg grid with a vertical legend at the bottom-left: each row is a (marker_color, fill_color) pair.
- Scattered L-shaped corner markers of each marker color define opposite corners of axis-aligned rectangles.
- Some markers are solid 2×2 blocks instead of corner pairs.

Rule:
- Read legend bottom-to-top; for each entry find all pixels of that marker color.
- Determine which two bbox corners carry L-shaped anchors — the *other* diagonal defines the fill region.
- Fill the diamond (all shortest-path cells between those two corners) with the legend's fill color.
- Solid 2×2 marker blocks are simply recolored to the fill color.
- Later legend entries are protected from earlier fills.

Key insight: The L-shaped anchors sit on the marker diagonal; the fill goes on the opposite diagonal of the bounding box.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="c4d067a0" lines="9-23">
<description>
Complete a grid of colored blocks to match a minimap legend in the top-left corner.

Input structure:
- bg = most common color.
- Top-left corner: a small sprite of single colored pixels forming a minimap pattern on a regular lattice.
- Main area: uniformly colored square blocks (NxN) arranged on a regular grid; some positions are empty (bg).

Rule:
- The legend pixels define a color pattern; each pixel's lattice position maps to a block position in the main grid via a fixed scale factor and offset.
- Existing colored blocks confirm the scale/offset alignment.
- For every legend pixel whose corresponding grid position is empty, place a new block of that color.

Key insight: The top-left pixel pattern is a scaled-down minimap of the block layout; the mapping is inferred from blocks already present.
</description>
</reference_puzzle>

你的任务：
以参考题为标杆，运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。用中文创造全新的 ARC 谜题描述。
创新方式包括但不限于：
**机制重组**：将多道参考题各自的核心机制提取出来，组合成一个新的多步管道，用到新的场景。
**机制变异**：取几道参考题的核心机制，以机制为基础进行创新并与其他多种机制进行创新重组，使解题路径发生质变而非量变。
**逆向构造**：取一道参考题的输出模式，思考"什么样的输入规则能产生类似的输出结构但需要完全不同的推理过程"。

严格使用下面的 XML 风格格式输出：

<new_puzzle>
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。
</new_puzzle>

请一次性生成 5 个 `<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 5 个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7. 题目难度应来自机制的隐蔽性而非步骤数。用参考题作为难度标尺——如果你生成的题一个样例就能猜到规则，说明机制太透明，需要加深隐蔽性。
8. 优先从参考题的机制出发进行重组或推演，而非从物理隐喻出发。物理隐喻容易导致大量题目归结为一种模式，例如"沿方向移动到阻挡处停止"。
9. 题目的难度优先来自规则的**结构层数**，而不是单一步骤的复杂度。关于"结构层数"的判定标准：**只有解题者需要主动推理才能完成的步骤才算一层**。肉眼一扫即可获得的信息（如"找到缺口在哪一侧"、"数出像素个数"、"识别哪个对象最大"）不构成独立的推理层——它们只是变换的前置输入，不增加认知难度。真正的多层结构是指每一层都需要解题者发现一个非显然的规律。以两层结构为例：
   - ✗ 伪两层："找到缺口方向"→"把东西推到对面" — 第一步肉眼可见，实际只有一层推理。
   - ✓ 真两层："通过孔洞数量匹配颜色图例"→"用匹配到的颜色重绘对象" — 两步都需要非平凡的归纳。

   如果你发现自己写出的题只是"对每个对象，根据它的某个属性做某变换"的单步规则，请引入一个需要主动推理的前置步骤。**用堆叠 tie-break 规则或复杂数学公式来"加难度"是劣解；用多一层需要归纳的结构嵌套来"加难度"才是好解**。
10. 在符合ARC题目和提示词要求的基础上，鼓励尝试创新规则为多层结构的题目，不局限于两层，可以创新三层或四层结构的题目，但不可为了多结构层数违背要求。

生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。没有使用上述注意事项第二条禁止使用的复杂计算。

