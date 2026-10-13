你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="7c66cb00" lines="9-24">
<description>
Dock two-tone sprites into color-matched horizontal bars with a color swap.

Input structure:
- bg = most common color.
- Top area: small two-tone sprites (outer color + inner color).
- Below: horizontal bars spanning most of the row, each with a fill color and border color on edges.

Rule:
- Match each sprite to the bar whose fill color equals the sprite's outer color.
- Place the sprite at its original x-position, aligned to the bar's bottom.
- Recolor: sprite outer → bar border, sprite inner → bar fill.
- Erase original sprites from the top area.

Key insight: Color matching is inverted — the sprite's dominant color identifies its target bar, then gets swapped to the bar's accent color.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="9ba4a9aa" lines="75-90">
<description>
Return the ring object reached by following the guide path from the checkerboard object.

Input structure
- The input has one dominant background color, one 3x3 checkerboard object,
  several 3x3 ring objects, and sparse guide pixels on a two-cell lattice.

Rule
- Find the checkerboard object and its unique adjacent guide pixel.
- Follow guide pixels of that color through two-cell horizontal/vertical steps.
- Output the unique 3x3 ring object adjacent to that connected guide path.

Key insight
- The checkerboard object is the source, and the color of its attached guide
  pixel selects the path whose far endpoint points at the answer ring.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="5623160b" lines="44-59">
<description>
Push every colored object touching maroon markers straight outward until it reaches the grid edge.

Input structure:
- The background is the dominant color.
- Maroon pixels are fixed markers.
- Every other foreground 4-connected object touches the marker pixels on exactly one cardinal side.

Rule:
- Clear each non-background, non-maroon object from its original position.
- Translate it in the direction away from its maroon contact until the object reaches the grid boundary.
- Keep the maroon marker pixels unchanged.

Key insight:
- The maroon contact side is a launcher: it determines a unique outward ray for the whole object, and the edge of the grid determines the translation distance.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="7666fa5d" lines="12-26">
<description>
Fill the parallelogram-shaped region between parallel anti-diagonal line segments with red.

Input structure:
- bg = most common color.
- One non-bg color forms short diagonal segments, each lying on a single anti-diagonal (r+c = const).
- Segments come in parallel pairs bounding a region to be filled.

Rule:
- For each main-diagonal axis (r−c = const), collect anti-diagonal segments whose span covers that axis.
- Sort those segments by their anti-diagonal index (r+c).
- Fill all bg cells lying strictly between consecutive segment pairs along that axis with red (color 2).

Key insight: The two diagonal directions decouple — segments sit on anti-diagonals, and filling sweeps along main diagonals between them, forming parallelogram regions.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="bc93ec48" lines="48-63">
<description>
Copy each foreground object anchored in a true grid corner clockwise into the next corner.

Input structure
- The dominant color is the background.
- Foreground objects are monochromatic 4-connected components.
- Relevant objects are exactly those whose bounding box touches two adjacent borders at one grid corner.

Rule
- Start from the input unchanged.
- For each corner object, copy its pixel mask and color to the next corner clockwise: top-left to top-right, top-right to bottom-right, bottom-right to bottom-left, and bottom-left to top-left.
- Preserve each object's orientation by aligning its bounding box to the destination corner; stamps may overwrite existing pixels at the destination, but source pixels are not cleared.

Key insight
- Interior and side-only objects are distractors; only components actually anchored to a corner participate in the clockwise perimeter copy.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="cf5fd0ad" lines="10-22">
<description>
Build a four-quadrant rotational tiling of the input square.

Input structure
- The input is a square grid treated as one complete sprite.

Rule
- Each output quadrant is a 2x2 repetition of a rotated copy of the whole input.
- The quadrant rotations are upper-left 180 degrees, upper-right 90 degrees
  clockwise, lower-left 90 degrees counterclockwise, and lower-right unchanged.

Key insight
- The output is a 4x4 grid of original-size blocks, not a per-pixel scale-up.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="7ec998c9" lines="14-28">
<description>
Draw a blue folded corner-to-corner route through the lone marker's column.

Input structure:
- A rectangular grid filled with one background color.
- Exactly one non-background marker pixel; blue is absent from the input.

Rule:
- Preserve the marker and repaint background pixels in its column blue from top to bottom.
- If the marker is in the unique center column, connect that column to the top-right and bottom-left corners.
- Otherwise, connect the column to the top-left and bottom-right corners.

Key insight:
The marker's column is the route spine; its central vs off-center placement selects which opposite corner pair gets the horizontal edge segments.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="ac0c2ac3" lines="7-19">
<description>
Fill each centered square ring with the color of the marker at that ring distance.

Input structure
The input is an odd square grid with a dominant background color and one
non-background marker for every Chebyshev distance from the grid center.
Rule
The output replaces the background with concentric square rings centered on
the middle cell; every cell at distance d from the center takes the color of
the unique marker whose distance from the center is d.
Key insight
The scattered markers are not objects to move; their Chebyshev distances
form a palette for the centered square rings.
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

<reference_puzzle index="10" source="train" puzzle_id="79cce52d" lines="29-44">
<description>
Crop off the control row and column, then cyclically roll the payload by the two marker offsets.

Input structure
- The grid is square, with a control row on top and a control column on the left.
- The control row and column share one filler color and one single-pixel marker color.
- The lower-right square is the payload and contains none of the control colors.

Rule
- Remove the top control row and left control column.
- Roll the remaining payload down by the marker's zero-based position in the left control column.
- Roll the payload right by the marker's zero-based position in the top control row.

Key insight
- The two border markers are cyclic shift amounts for the payload, not colors to preserve.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="7b0280bc" lines="54-69">
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

<reference_puzzle index="12" source="eval" puzzle_id="35ab12c3" lines="73-88">
<description>
Complete outlined shapes and carry attached shadow pixels along their edges.

Input structure:
- bg = background. Each active color marks vertices of an 8-direction aligned outline.
- A start vertex can have adjacent non-bg, different-colored shadow pixels.
- Other colored pixels may be distractors unless they lie on valid drawable edges.

Rule:
- For each color, order its pixels by polar angle around their centroid to form a cycle.
- Draw only horizontal, vertical, or 45-degree edges whose interiors are clear.
- From the chosen start vertex, copy adjacent shadow offsets along each drawable edge
  when the parallel path is in bounds and clear.

Key insight: The main outline is reconstructed first, then the start vertex's neighbor offsets define parallel shadow strokes.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="f560132c" lines="259-273">
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

<reference_puzzle index="14" source="train" puzzle_id="902510d5" lines="58-80">
<description>
Keep the largest 8-connected motif and replace the corner/control pixels with a solid corner triangle.

Input structure
- The background is the dominant color.
- Exactly one grid corner contains a foreground control pixel; it gives the
  corner from which the triangle is painted.
- There is one unique largest 8-connected monochrome foreground object, the
  motif that remains in the output; all other foreground pixels are controls.
- Every foreground component outside that motif is a singleton control pixel.
- The fill color is the unique control color that appears in at least two
  non-corner control pixels.

Rule
- Start from a blank background canvas and redraw only the largest motif.
- Count the non-corner control pixels; that count is the triangle side.
- Paint the fill color on the right-triangle region where those two distances
  from the marked corner's edges sum to less than the side length.

Key insight
- The scattered singleton controls are counted into a compact triangular
  stack at the marked corner; only the largest motif is not part of the count.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="42f83767" lines="23-39">
<description>
Expand the lower color grid by replacing each pixel with its matching top stencil.

Input structure
The input has a top legend and a lower dense color grid, separated by blank rows.
The legend contains left-to-right 2x2 color swatches and, to their right,
the same number of square neutral-color stencil masks.

Rule
Pair the swatches and stencil masks by left-to-right order. For every pixel in
the lower color grid, paste the paired stencil mask as one output block,
recoloring the mask pixels with that pixel's color and leaving mask holes as bg.

Key insight
The lower grid is not copied directly; it is an index grid whose colors select
the stencil shape from the legend, so each input pixel becomes one square block.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="e12f9a14" lines="138-152">
<description>
Emit diagonal beams from framed 2x2 blocks through frame gaps; beams interact on collision.

Input structure:
- bg = most common color. Wall color (e.g. green) forms 4x4 frames around 2x2 monochrome blocks.
- Each frame has one or more gap pixels where the wall is missing.

Rule:
- Each gap emits a beam of the inner block's color, traveling diagonally outward (direction set by gap position relative to the block).
- Beams paint every cell they cross until they hit the grid edge or collide.
- 90° collision: both beams turn 45° toward their bisector direction; the intersection cell is left blank (1-pixel gap).
- 45° collision: beams merge into a zigzag (alternating cardinal + diagonal steps) with no gap at the meeting point.

Key insight: Two distinct collision rules — 90° produces a gap and single-pixel diagonal lines; 45° produces no gap and thicker zigzag bands.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="e4888269" lines="67-88">
<description>
Recolor every target pixel by following the ordered two-column substitution legend.

Input structure
- A dominant background surrounds one multicolor vertical legend: a connected,
  fully filled two-column block with one color pair per row.
- Other non-background pixels or objects are the targets; divider or frame
  pixels are ordinary target-side structure that remain unchanged unless their
  color is rewritten by the legend.

Rule
- Read the legend rows from top to bottom as substitutions from the left
  column color to the right column color.
- For each non-background pixel outside the legend, scan those rows in order;
  whenever the current color equals the row's left color, replace it with the
  row's right color and keep scanning lower rows.
- Leave the legend itself and background pixels unchanged.

Key insight
- The same color may be rewritten multiple times, but only by substitutions
  that appear later in the vertical legend, so the row order is part of the
  symbol instruction.
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

<reference_puzzle index="19" source="train" puzzle_id="33067df9" lines="32-48">
<description>
Expand the sparse dot lattice into a 26x26 drawing with thick bars between adjacent matching dots.

Input structure
- The background is the dominant color.
- Foreground pixels lie on logical rows and columns separated by background spacer rows/columns.
- A logical row or column is present when it contains at least one foreground pixel.

Rule
- Expand every background spacer row/column to thickness 2.
- Expand the logical rows/columns equally so the full output is 26x26.
- Paint each foreground logical pixel as a solid rectangle of its color.
- Connect adjacent same-color logical pixels horizontally through the intervening spacer.
- Connect adjacent same-color logical pixels vertically only when neither endpoint is part of a horizontal same-color run.

Key insight
- The input is a sampled line drawing: horizontal same-color runs define the bars first, and only unclaimed same-color vertical neighbors are joined.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="d749d46f" lines="32-51">
<description>
Pack each solid foreground rectangle twice: wide across the top and tall from the bottom.

Input structure
The grid has one background color and one foreground color; the foreground forms
separate solid rectangles, ordered from left to right.

Rule
Build a 10-row background canvas whose width is the packed width of the wide
versions of the rectangles. For each source rectangle, paint a top-aligned
rectangle of shape min(h,w) by max(h,w), with one background column between
consecutive rectangles. Then, starting again from the left edge, paint a
bottom-aligned rectangle of shape max(h,w) by min(h,w) for each source rectangle,
again separated by one background column.

Key insight
The input rectangles are not copied in their original orientations; each object
contributes one horizontal-facing and one vertical-facing normalized rectangle,
preserving left-to-right object order.
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

