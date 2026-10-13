你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="b2bc3ffd" lines="8-23">
<description>
Move every object above the solid bottom stripe upward by its own pixel count.

Input structure
The grid has a dominant background, exactly one full-width non-background row
that acts as a fixed stripe, and one or more 4-connected monochrome objects
above that stripe.

Rule
Clear each object from its original cells, then translate it straight upward
by the number of pixels in that object, preserving its shape and color. Leave
the full-width stripe unchanged.

Key insight
Each object's area is also its vertical displacement; the full-width row is a
fixed baseline, not a moving object.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="2601afb7" lines="30-45">
<description>
Rotate each vertical bar's color one slot right and its height one slot left.

Input structure
- A dominant background surrounds several monochrome one-cell-wide vertical bars.
- Every bar is bottom-aligned and occupies a distinct column.
- The ordered bar slots are read from left to right.

Rule
- Keep the grid shape and the bar columns unchanged.
- At slot i, use the color from the previous slot cyclically.
- At slot i, use the height from the next slot cyclically.

Key insight
- Color and height are independent cyclic attributes of the ordered bar slots:
  colors rotate right while heights rotate left.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="d8e07eb2" lines="29-47">
<description>
Look up query glyphs in a reference table and mark whether they form a line.

Input structure:
- Teal background with two full-width pink separator rows.
- Above the first separator is a query row of 5x5 glyph cells.
- Between separators is a reference table of the same 5x5 cells; a glyph is
  identified by both its color and its 3x3 shape.

Rule:
- Green-fill the background inside every table cell whose glyph exactly
  appears in the query row.
- If the matched table cells form one complete row or one complete column,
  also green-fill the query band and the bottom verdict band.
- Otherwise leave the query band unchanged and fill the bottom verdict band red.

Key insight: The bottom color is not a count; it is the yes/no answer to
whether the queried glyphs align as a full row or column in the table.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="4e7e0eb9" lines="31-49">
<description>
Transform each 9x9 panel by mirroring across its center guide line or filling all four block slots from its lone exemplar.

Input structure
The grid is made of 9x9 panels separated by one-cell-wide grid lines. Each
panel has four monochrome 3x3 slots at the quadrant positions, with either a
full non-background center row, a full non-background center column, or no
guide line.

Rule
A panel with a full center column is reflected left-to-right, and a panel
with a full center row is reflected top-to-bottom. A panel with no guide line
has exactly one slot color that appears once and one slot color that appears
three times; recolor all four slots to the once-occurring slot color.

Key insight
The center guide line encodes the mirror axis, while panels without a guide
line are completed by copying the structurally unique slot color.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="a6953f00" lines="45-63">
<description>
Extract the top-left or top-right 2x2 corner crop selected by the top edge clue.

Input structure
- The input is a square color grid whose answer is one of the two 2x2 crops
  touching the top edge and a grid corner.
- A repeated color on the top row, when present, acts as the primary side cue.
- Otherwise, the two top corner colors have different whole-grid frequencies.

Rule
- If a top-row color repeats, find the repeated top-row occurrence nearest to
  a left or right edge and take the 2x2 top-corner crop on that side.
- If the top row has no repeated color, compare the whole-grid frequencies of
  the two top-corner colors and take the side whose corner color is rarer.

Key insight
- The task is not asking for an arbitrary 2x2 search: the decisive information
  is on the top edge, and it selects which top corner crop to keep.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="320afe60" lines="41-56">
<description>
Slide each foreground object horizontally to the edge indicated by whether its shape has an open notch.

Input structure:
- Yellow is the background and there is one blue foreground color.
- The foreground is split into separated 4-connected objects.
- Each object keeps its row span and shape in the output.

Rule:
- If an object's bbox border contains a background gap flanked by object pixels along that same border, slide it to the right edge and recolor it green.
- Otherwise slide the object to the left edge and recolor it red.
- Start from a blank background canvas; no moved objects overlap.

Key insight:
- The target side is encoded by the object's shape, not by its original position: open notches go right, while closed or solid shapes go left.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="6165ea8f" lines="74-90">
<description>
Build a color-labeled shape-equivalence table from the objects listed on the right edge.

Input structure:
- The background contains one main 4-connected object for each foreground color.
- The rightmost column has one singleton pixel of each foreground color, listed top-to-bottom.
- Those singleton pixels define the row and column order of the output table.

Rule:
- Normalize each main object to its tight mask and compare masks up to rotation or reflection.
- Draw 2x2 color labels across the top and down the left, separated by one background row/column.
- For each off-diagonal table cell, paint a 2x2 red block when the two labeled objects have the same shape under rotation/reflection, otherwise paint a 2x2 grey block.
- Leave diagonal cells and all separators as background.

Key insight:
- Object positions and original orientations are distractors; the relation is dihedral shape equivalence, while the right-edge legend fixes the table order.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="ad38a9d0" lines="23-36">
<description>
Recolor each placeholder object according to its normalized shape while leaving the background fixed.

Input structure
A same-size grid with one dominant background color and one foreground
placeholder color forming separated 4-connected objects.

Rule
Replace every placeholder object by the color assigned to its exact
translation-normalized mask; all background pixels remain unchanged.

Key insight
The input objects are a shape legend with repeated instances: identical
masks always receive identical colors, independent of their position.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="a416fc5b" lines="95-106">
<description>
Add the two missing colored arms of a centered diamond-sprite motif, or clear a completed motif to a blank square.

Input structure
An 11x11 board has orange background and pink separator bars forming nine 3x3 cells. Occupied cells contain a four-pixel diamond sprite; the center cell and one neighboring cell are red, and a completed board may also contain one grey and one teal sprite.

Rule
With only the two red sprites present, preserve the board and add grey and teal diamond sprites 135 degrees clockwise and counterclockwise from the non-center red arm. If those expected grey and teal sprites are already present, output an all-background square whose side length is the total number of sprite pixels.

Key insight
The non-center red sprite is an eight-neighbor direction from the center; the other two colored arms are fixed rotations of that direction.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="d753a70b" lines="81-93">
<description>
Red diamond outlines contract by one layer and gray diamond outlines expand by one layer while other objects stay fixed.

Input structure
- A 16x16 orange-background grid contains one-cell-thick Manhattan-diamond outlines.
- Red and gray outlines are the changing objects; other colored diamond outlines are static.
- Some outlines may be clipped by the grid boundary, but their visible pixels still fit a clipped Manhattan-distance contour.
Rule
- Clear all red and gray pixels.
- Red objects are redrawn with the same diamond center and radius minus one.
- Gray objects are redrawn with radius plus one; an ambiguous edge-clipped growing arc continues away from the nearest perpendicular edge.
Key insight
- The task is not about moving whole objects: each changing object is a contour of a diamond, and the visible operation is one step of contour contraction or expansion.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="7acdf6d3" lines="25-42">
<description>
Move the source-color pixels into the unique outline interior whose size matches the source count.

Input structure
- The grid has a dominant background and exactly two foreground colors.
- One foreground color forms 8-connected outline objects with row-wise interior cells.
- The other foreground color supplies a set of source pixels away from the chosen outline.
Rule
- For each possible outline color, compute each 8-connected outline object's
  row-span interior: background pixels strictly between the leftmost and
  rightmost outline pixels on each occupied row.
- Select the unique outline interior whose cell count equals the total
  number of pixels of the other foreground color.
- Erase the source pixels and paint that matched interior with the source color.
Key insight
- The colors are roles, not fixed IDs: the source color is identified by
  counting, and the destination is the outline with a matching interior area.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="42f83767" lines="23-39">
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

<reference_puzzle index="13" source="train" puzzle_id="25c199f5" lines="30-44">
<description>
Pack the non-background rows from three separated panels into one panel, scanning panels right-to-left.

Input structure:
- A rectangular grid contains three equal-size panels separated by two full-height divider columns.
- The background is the dominant color; divider pixels use a color that appears only in the divider columns.

Rule:
- Read panels from right to left, and within each panel read rows from top to bottom.
- Keep each row that contains any non-background pixel, discarding all-background rows.
- Place the kept rows at the bottom of a single panel-shaped output, padding above with background rows.

Key insight:
- The task is row packing: the divider columns mark independent panels, and only the occupied rows are moved.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="396d80d7" lines="18-30">
<description>
Paint the inner foreground color into every background pixel that touches the enclosing frame only diagonally.

Input structure:
- The grid has one dominant background color and exactly two foreground colors.
- One foreground color is an enclosing frame: its bbox strictly contains the bbox of the other, inner color.

Rule:
- Keep all existing pixels unchanged.
- For each background pixel, if it is diagonally adjacent to the frame color and has no orthogonal adjacency to the frame color, recolor it to the inner color.

Key insight: The inner color marks the color to use, while the frame's diagonal-only background neighbors define the cells to add.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="d59b0160" lines="39-54">
<description>
Delete connected components whose colors match a legend swatch.

Input structure:
- Background (bg) fills most of the grid.
- Top-left corner: a small square legend region bordered on its bottom and right edges by a single color; the interior holds a set of "target" colors.
- Elsewhere: multicolored objects (4-connected components) on the bg.

Rule:
- Read the target color set from inside the legend border.
- For each connected component, check whether it contains **all** target colors.
- Erase (fill with bg) every component that does; leave all others unchanged.
- Output grid is the same size.

Key insight: The legend acts as a filter specification — only components carrying the full color palette shown in the legend are removed.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="e734a0e8" lines="44-59">
<description>
Copy the unique colored sprite tile into every tiled cell marked by an internal separator-colored pixel.

Input structure
- Full same-color separator rows and columns partition the grid into equal rectangular tiles.
- One tile contains the only non-background, non-separator colored sprite.
- Target tiles are otherwise background-colored but contain at least one internal pixel of the separator color.

Rule
- Keep the separator lines and the source sprite tile unchanged.
- Replace each marked target tile with an exact copy of the source tile.
- Unmarked background-only tiles remain unchanged.

Key insight
- The internal separator-colored pixels are placement markers; the single colored tile is the template to stamp into those marked cells.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="64efde09" lines="68-83">
<description>
Copy colored control-ray ports from one rectangular sprite orientation to every matching transformed sprite.

Input structure:
- bg is the most common color.
- Several dense 2-by-N or N-by-2 rectangular sprites are dihedral transforms of one template.
- Isolated single-pixel control markers sit on rays projected outward from the dominant long edge of some sprites.

Rule:
- For each sprite, the active edge is the long side whose cells have the strongest same-color majority.
- A control marker seen first along an outward ray from an active-edge cell assigns its color to that template cell.
- On every transformed sprite, draw that assigned color outward from the corresponding active-edge cell through bg cells until blocked or out of bounds.

Key insight:
- The marker positions are not absolute; they label cells of a shared rectangular template, then the labels follow each sprite's rotation/reflection.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="f8cc533f" lines="10-25">
<description>
Complete every colored sprite by applying the union of all aligned partial sprite masks.

Input structure:
- The dominant color is the background.
- Each non-background color appears as a separated partial copy of one common sprite mask.
- The copies share a top-left-aligned sprite window; some copies may be missing lower or right edge pixels.

Rule:
- Crop each same-color mask to its bounding box.
- Pad the crops to the maximum observed height and width, aligned at the top-left, and OR them into one complete template.
- Fill that template at every sprite location using the sprite's own color.

Key insight:
- No single input sprite is complete, but their missing cells are complementary, so the aligned union reveals the full mask.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="68bc2e87" lines="46-60">
<description>
Recover the bottom-to-top stacking order of overlaid rectangular color frames.

Input structure:
- The background is one color, and every foreground color is a one-cell-thick rectangular frame.
- Overlapping frame pixels show only the color of the frame that is highest at that location.

Rule:
- Reconstruct each frame as the border of the bounding box of its visible pixels.
- At every cell covered by multiple reconstructed frames, the visible frame color is above the other frames covering that cell.
- Return the unique bottom-to-top order of frame colors as a single-column grid.

Key insight:
The hidden parts of each frame do not need to be seen directly; the visible pixels reveal the full frame bbox, and overlap cells reveal the layer constraints.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="b7256dcd" lines="18-32">
<description>
Transfer each colored marker to its adjacent neutral mask component, then erase the markers.

Input structure
- The background is the most common color.
- The neutral mask color is the unique most frequent foreground color.
- Other foreground colors are marker pixels that may touch neutral 4-connected mask components.
Rule
- Remove every marker pixel by recoloring it to the background.
- For each 4-connected neutral mask component, find marker colors touching it by edge adjacency.
- If a component touches one marker color, recolor the whole neutral component to that marker color.
- If a component touches no marker color, leave it neutral.
Key insight
- Marker pixels label neighboring neutral regions but are not part of the final colored objects.
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
   - **不建议使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id是不可用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
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
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。
