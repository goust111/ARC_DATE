你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="db695cfb" lines="8-19">
<description>
Connect same-color diagonal pixel pairs, then emit perpendicular rays from differently-colored pixels on those diagonals.

Input structure:
- bg = most common color. Line color (e.g. blue) has pixel pairs aligned on shared diagonals. Trigger color (e.g. magenta) pixels sit along those diagonals.

Rule:
- Draw diagonal lines connecting each pair of same-color aligned pixels.
- Where the diagonal crosses a trigger pixel, emit perpendicular diagonal rays (in the trigger's color) extending to the grid boundary.

Key insight: Trigger pixels act as beam-splitters — the perpendicular rays rotate 90° from the connecting diagonal's orientation.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="342dd610" lines="18-36">
<description>
Move every foreground pixel by the fixed offset indicated by its color.

Input structure:
- The grid has a unique dominant background color.
- The foreground is a sparse set of colored pixels.
- Each foreground color is one of the trained motion colors: orange, red,
  maroon, or blue.

Rule:
- Clear the canvas to the background color.
- Translate orange pixels up 2 cells, red pixels left 2 cells, maroon
  pixels down 2 cells, and blue pixels right 1 cell.
- Preserve each moved pixel's original color.

Key insight:
The foreground color is the motion instruction; position only determines
where that color-specific displacement lands.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="a644e277" lines="31-50">
<description>
Extract the lattice subregion bounded by the four background-colored corner holes.

Input structure:
- A regular one-cell-thick lattice has one line color, one background color,
  and sparse marker pixels.
- Exactly four lattice-line intersections are background-colored holes instead
  of line-color pixels.

Rule:
- Find the unique non-background color that forms regular horizontal and
  vertical lattice lines.
- Use the four background-colored lattice intersections as the corners of one
  rectangle.
- Return the inclusive crop between those corner rows and columns.

Key insight:
- The crop is marked by the missing lattice intersections, not by the marker
  pixels inside the cells.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="e048c9ed" lines="11-26">
<description>
Add a quantity-color label in the top marker column for each horizontal bar.

Input structure
- The grid has a black background and one non-background marker pixel in the top row.
- Below the marker row, every non-empty source row contains one solid monochrome horizontal bar to the left of the marker column.
- Source rows are staggered every other row.

Rule
- Copy the input grid.
- For each source bar, read its quantity as the bar length.
- Paint the quantity color in the marker column on the same row as the bar:
  2->1, 3->4, 4->9, 5->6.

Key insight
- The grey marker supplies the result column, while each bar supplies a small quantity with a fixed color code.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="e39e9282" lines="73-81">
<description>
Project maroon markers into adjacent grey or pink 3x3 squares, erasing grey squares afterward.

Input structure
The grid has a dominant background, solid 3x3 grey and/or pink squares, and maroon marker pixels or segments directly touching one side of exactly one square.
Rule
Grey squares are erased; their adjacent maroon markers stay in place and stamp the nearest edge cell inside the grey square. Pink squares stay pink; their adjacent maroon markers disappear and stamp the corresponding cell on the square's middle row or middle column.
Key insight
The side touched by a marker determines the projection: grey uses the near edge, while pink uses the centerline, preserving the marker's coordinate along that side.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="ad3b40cf" lines="72-87">
<description>
Mirror the smallest non-axis foreground color across the full straight axis.

Input structure
- The background is the dominant color.
- One foreground color forms exactly one full row, column, main diagonal, or anti-diagonal reflection axis.
- The remaining foreground colors include a unique least-populated color to copy.

Rule
- Preserve the input grid.
- Select every pixel of the unique smallest non-axis foreground color.
- Reflect those pixels across the full straight axis and paint the reflected positions with the same color.

Key insight
- The axis color marks the mirror line, while the color population chooses which set of objects is duplicated.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="18286ef8" lines="17-31">
<description>
Use the pink control pixel to choose where the maroon pixel belongs inside the grey 3x3 region, and turn the control pixel maroon.

Input structure:
- The grid contains one 3x3 region made from eight grey pixels and one maroon pixel.
- There is exactly one pink control pixel outside that region.

Rule:
- Restore the old maroon cell in the 3x3 region to grey.
- Compare the pink pixel to the 3x3 region: above/same band/below selects top/middle/bottom, and left/same band/right selects left/middle/right.
- Paint the selected cell maroon and recolor the pink control pixel maroon.

Key insight:
- The pink pixel acts as a relative-position marker for a 3x3 target inside the grey region.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="f560132c" lines="259-273">
<description>
Assemble scattered jigsaw pieces into a rectangle, recoloring each by a legend sprite's color map.

Input structure:
- bg = most common color.
- Several monochrome non-bg sprites (the "pieces"), scattered across the grid.
- One sprite is the legend: it contains a multi-color k×k patch (typically 2×2) encoding output colors.

Rule:
- Tile all pieces (rotations/reflections allowed) into a tight rectangle with no gaps.
- The legend's k×k color patch defines a spatial color map over the rectangle.
- Each piece is recolored to the legend-map color corresponding to its position in the assembled rectangle.

Key insight: The legend piece doubles as both a jigsaw piece and a color key — its internal k×k palette assigns colors by spatial quadrant.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="97d7923e" lines="81-94">
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

<reference_puzzle index="10" source="train" puzzle_id="dce56571" lines="8-23">
<description>
Straighten the single foreground mask into a centered horizontal segment of the same color and length.

Input structure:
- The grid has one dominant background color.
- All non-background pixels share one foreground color.
- The number of foreground pixels fits in one centered row of the grid.

Rule:
- Erase the original foreground mask.
- Count its pixels, keep its color, and paint that many pixels as a horizontal segment.
- Place the segment on the middle row with equal background margin on the left and right.

Key insight:
- The original foreground shape is irrelevant except for its color and area; the output is the area normalized into the grid's central horizontal line.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="1e5d6875" lines="26-45">
<description>
Paint a recolored diagonal copy from each L triomino source object.

Input structure:
- The background is the dominant color.
- The only foreground objects are 4-connected red and grey L triominoes,
  each occupying three pixels of a 2x2 box.

Rule:
- Grey L objects paint a yellow copy translated one diagonal step toward
  their missing 2x2 corner.
- Red L objects paint a green copy translated one diagonal step away from
  their missing 2x2 corner.
- Copies are clipped by the grid boundary; red-derived green copies are
  painted after grey-derived yellow copies where the projections collide.

Key insight:
- The missing corner of each 2x2 L object identifies its diagonal axis, and
  the source color determines which side of that axis receives the copy.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="eee78d87" lines="40-55">
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

<reference_puzzle index="13" source="train" puzzle_id="57edb29d" lines="88-103">
<description>
Extract the single plain rectangle and redraw it with the aligned rectangles' border-relative marker pattern.

Input structure:
- The boundary color is the background/separator.
- Foreground objects are dense rectangular regions; exactly one rectangle has only its base color.
- Other rectangles contain their base color plus marker pixels.

Rule:
- Crop the only plain rectangle; this crop is the output canvas.
- Use marked rectangles whose row span overlaps the plain rectangle's row span.
- Transfer their marker pixels by preserving each marker run's distance from the nearest rectangle edges.

Key insight:
- The marked rectangles are examples of an edge-relative pattern, so the missing plain rectangle receives the same pattern resized to its own dimensions.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="7e2bad24" lines="91-106">
<description>
Extend the diagonal blue ray through the grid, reflecting it off red line-segment mirrors.

Input structure
- Black-background grid with one diagonal 8-connected blue ray seed.
- Red foreground objects, when present, are straight horizontal or vertical mirror segments.
- Other foreground bars are transparent to the ray and may be overwritten where the ray crosses them.

Rule
- Start from the ray seed endpoint whose outward diagonal extension remains inside the grid.
- Paint the ray continuously in that diagonal direction until it leaves the grid.
- If the next cell is a red mirror, do not paint it; reflect by flipping the row direction on a horizontal mirror or the column direction on a vertical mirror.

Key insight
- The non-red bars are decoys: only red bars reflect the diagonal path, while all other crossed cells become blue.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="7d7772cc" lines="70-92">
<description>
Move each source pixel toward the separator when it matches its aligned reference pixel, otherwise to the far edge.

Input structure:
- A single full separator line is part of a rectangular frame; the frame's
  extra cells mark the reference side.
- On the reference side, one row or column of colored pixels is aligned
  with one row or column of source pixels on the opposite side.
- Each side has its own dominant background color.

Rule:
- Compare each source pixel with the reference pixel on the same row or column.
- If their colors match, place the source pixel in the cell adjacent to the
  separator on the source side.
- If their colors differ, place the source pixel on the far outer edge along
  the same row or column.
- Erase the original source pixels to the source-side background; keep the
  frame and reference pixels unchanged.

Key insight:
The extra frame cells identify which side is the reference; equality with the
aligned reference pixel decides whether each source pixel moves inward or outward.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="16de56c4" lines="9-34">
<description>
Fill periodic progressions that shoot from a unique grid edge.

Input structure:
- bg = most common color.  Sparse non-bg dots of several colors.
- All repeating colors originate from the same grid edge (top, bottom,
  left, or right — the one with the most non-bg dots).
- Each repeating color has exactly 2 dots per shooting-direction line;
  their gap defines the step.
- Every non-bg dot lies on a shooting-direction line that contains a
  repeating color.

Rule:
- Determine the unique shooting edge; left/right → rows, top/bottom → columns.
- For each shooting-direction line, the repeating color has exactly 2 dots;
  step = gap between them, offset = first position mod step.
- A different-colored dot aligned to the same beat (pos % step == offset) is
  a companion: it absorbs the ray and becomes the fill color, limiting the
  fill range to the span between repeating and companion dots.
- An unaligned dot is a bystander and left untouched.
- Without a companion the repeating color fills the entire line.

Key insight: the shooting edge is identified by counting boundary dots;
the gap between the two marker dots is the period; a companion sharing the
modular class acts as a wall that stops the ray and dictates the paint color.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="7b5033c1" lines="7-21">
<description>
Straighten a colored snake path into a single-column strip preserving pixel order.

Input structure:
- bg = most common color.
- Non-bg pixels form a single 4-connected path (snake) with exactly two endpoints.
- Each path pixel may be a different non-bg color.

Rule:
- Find the two endpoints (cells with only one path neighbor) of the snake.
- Begin at the top-left-most endpoint and trace the path cell by cell.
- Output an (N×1) column listing each cell's color in traversal order.

Key insight: The path is a simple chain with no branches; choosing the top-left-most endpoint fixes a unique reading direction.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="a1aa0c1e" lines="74-97">
<description>
Compress the stacked colored bars into a compact ordered table with the control pixel on the selected row.

Input structure
- A background grid contains several full-width colored bars from top to
  bottom; the last bar is the base/separator color.
- Each non-base bar may have an aligned three-column comb motif below it,
  alternating side-pair rows with solid rows.
- A singleton control pixel sits away from the colored motifs.

Rule
- Make one output row per non-base bar, preserving top-to-bottom order.
- In that row, paint the bar color once for each solid three-pixel row in
  its motif, then paint the base-bar color in the separator column.
- The singleton control pixel sits immediately above the base bar. Among
  motifs whose open side-pair parity is compatible with that bottom row,
  put the control pixel in the output row with the fewest solid rows; use
  horizontal distance to the motif center only to break count ties.

Key insight
- The vertical comb height is a count, while the control pixel is read by
  projecting it onto the center gap of compatible comb rows rather than by
  its vertical band alone.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="458e3a53" lines="19-32">
<description>
Extract the compact table of solid cells hidden inside a separator lattice.

Input structure
- The grid has full-width and full-height bars of one separator color, dividing it into rectangular cells.
- Most cells contain a repeating multicolor pattern, while a contiguous rectangle of cells is solid monochrome.

Rule
- Split the input into lattice cells using the full separator rows and columns.
- Keep the cells whose full interior is one color, crop to their occupied rectangle, and output those colors.

Key insight
- The periodic pattern is noise; only completely solid lattice cells encode the reduced output grid.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="898e7135" lines="7-20">
<description>
Inflate a frame's hole-layout and fill each hole with its matching piece.

Input structure:
- bg = most common color. Largest component is a monochrome frame enclosing bg holes.
- Outside pieces composed of k×k blocks; k = side of smallest solid square piece.

Rule:
- Extract connected hole regions inside the frame as slots.
- Match each piece to a slot by comparing block-level shapes under rotation/flip.
- Output the layout scaled by k, with pieces placed in slots; frame color as background.

Key insight: Pieces are k×k-block-scaled versions of holes; downsampling reveals shape matches under all 8 orientations.
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

请一次性生成5个`<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
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

