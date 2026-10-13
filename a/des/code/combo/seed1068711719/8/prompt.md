你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="135a2760" lines="6-19">
<description>
Repair noisy periodic patterns inside frame-delimited grid regions.

Input structure:
- Frame color (most common) forms full rows/cols, dividing the grid into rectangular regions.
- bg (second most common) fills empty space; each region contains a small colored tile repeated periodically with a few corrupted pixels.

Rule:
- For each region, find the bounding box of non-bg content.
- Infer the smallest repeating tile that explains the pattern (majority vote per residue position), requiring it repeat at least 2x per axis.
- Replace the region with the clean tiled version.

Key insight: Tile dimensions are unknown and must be inferred; prefer the smallest tile with genuine repetition over one that memorizes the entire region.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="78e78cff" lines="71-91">
<description>
Recolor the row-wise passage indicated by wall pixels with the singleton marker color.

Input structure:
- A dominant background contains one singleton marker pixel and one wall color.
- Wall pixels lie on several rows and form left/right limits around the marker column.
- Empty rows between the top and bottom wall rows are open horizontal passages.

Rule:
- On a wall row, paint background pixels between the nearest wall strictly
  left of the marker column and the nearest wall at or right of it; if only
  one side is present, paint from that wall to the opposite grid edge.
- On empty rows between the top and bottom wall rows, paint the whole row.
- On empty rows outside the wall-row band, reuse the nearest end wall-row interval.
- Preserve wall pixels and the original marker pixel.

Key insight:
- The wall color defines scanline limits relative to the marker column; the
  marker color fills those implied horizontal spans rather than flood-filling
  through connected background.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="f3e14006" lines="21-39">
<description>
Build the woven rectangle determined by the two marked cells on each crossed bar.

Input structure
A black-background grid contains one long horizontal bar and one long vertical bar.
Each bar has a dominant base color and exactly two same-colored marker pixels.

Rule
The horizontal marker pair gives the output column span.  From the horizontal bar
row, walk toward the farther vertical marker; rows and columns alternate every
other cell.  Even step/even offset cells use the horizontal marker color until
the nearer vertical marker is reached, then use the vertical marker color.
Odd row steps use the horizontal base on marker columns and the original
crossing color on spacer columns.  Spacer columns on even row steps use the
vertical base, except at the switch row where they use the crossing color.

Key insight
The original cross cell records which base color wins at base/base lattice
crossings; the nearer vertical marker is the color-switch boundary.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="758abdf0" lines="54-69">
<description>
Advance short marker bars away from the full edge bar, turning completed two-pixel markers into edge-color marks at the opposite side.

Input structure
- The background is the dominant color.
- One non-background color forms a complete one-cell-thick bar on exactly one outer edge.
- The other foreground color appears only in the first two cells inward from that bar, lane by lane.

Rule
- In each lane perpendicular to the edge bar, a one-pixel marker adjacent to the bar grows one more cell inward.
- A two-pixel marker in those first two inward cells is cleared back to background.
- The cleared two-pixel marker is replaced by a two-pixel mark of the edge-bar color at the opposite side of the same lane.

Key insight
- The full edge bar defines both the orientation and the direction of the local two-step marker cycle.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="34cfa167" lines="94-117">
<description>
Complete a four-corner framed rectangle from one decorated anchor rectangle and its diagonal mate.

Input structure:
- The background is the dominant color.
- Two same-color solid rectangular anchors of the same size sit on opposite
  diagonal corners, with the first anchor above-left of the second.
- The upper-left anchor has two one-cell-thick vertical control bars to its
  right and two one-cell-thick horizontal control bars below it.

Rule:
- Copy the anchor rectangle into the two missing corners of the implied
  larger rectangle.
- Repeat the right-side control bars horizontally between the left and right
  anchors, using their spacing as the period, and draw the outer top and
  bottom borders in the first right-control color.
- Repeat the below-side control bars vertically between the top and bottom
  anchors, using the same period, and draw the outer left and right borders
  in the first below-control color.

Key insight: The small control bars are the first two ticks of two periodic
frame patterns; the diagonal anchor supplies the opposite corner and the
missing two anchor copies.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="6cbe9eb8" lines="79-94">
<description>
Rebuild the foreground rectangles as a largest-to-smallest lower-left nesting.

Input structure
- A periodic wallpaper background is overlaid by foreground colors that each form a rectangular frame or solid rectangle.
- The largest foreground rectangle is the output frame; smaller foreground rectangles are the objects to nest inside it.
- A frame may be partially hidden by a smaller rectangle, but its visible pixels still determine its bbox.

Rule
- Ignore the wallpaper and crop the result to the largest foreground rectangle's bbox size.
- Sort foreground rectangles by bbox area from largest to smallest.
- Reconstruct each frame or solid rectangle, placing every next smaller rectangle in the lower-left of the current rectangle's usable region.

Key insight
- The scattered background colors are not objects; the meaningful order is the unique descending rectangle size chain, and each step nests into the previous rectangle's lower-left interior.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="15660dd6" lines="33-54">
<description>
Combine each column's repeated template mask with its single solid color swatch.

Input structure:
- The grid is a table of framed panels separated by background rows/columns.
- The leftmost column gives one solid row-label color for each panel row.
- In each panel column, exactly one panel has a solid non-background interior,
  while the other panels share the same mask drawn in one template color on
  the background.

Rule:
- For every panel column, find the row containing the solid interior swatch.
- Rebuild one output panel using that row's left label as the frame color.
- Fill the repeated template mask with the swatch color and leave the mask
  background cells as background; separate rebuilt panels with one
  background column.

Key insight:
The table encodes a colorized stencil per column: the solid panel supplies
the paint color and frame label, while the duplicate template panels supply
the shape mask.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="5adee1b2" lines="8-23">
<description>
Use the left-edge color legend to draw a padded rectangular halo around each matching object while preserving enclosed holes.

Input structure:
- The grid has a dominant background color.
- One or more two-row markers sit on the left edge; each marker is a 2x2 block whose left column is an object color and right column is its halo color.
- Away from the markers, every source color from the legend appears as one or more 4-connected monochrome objects.

Rule:
- Keep the input pixels unchanged.
- For each source-color object, take its tight bbox expanded by one cell and clipped to the grid.
- Recolor background pixels in that expanded region to the marker's halo color, except for background holes enclosed by the object.

Key insight:
- The small left-edge bars are a legend: they map object colors to the colors used for the surrounding padded bbox.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="9f8de559" lines="19-37">
<description>
Erase the first border pixel hit by extending the red marker away from its pink trail.

Input structure
- The dominant background surrounds a unique red marker and a four-pixel
  pink trail.
- The pink pixels form one straight 8-connected ray starting next to the
  red marker.
- In the opposite direction from that trail, the red marker faces a
  non-background border or obstacle pixel beyond background cells.

Rule
- Keep the whole grid fixed except for the first non-background pixel on
  the ray from the red marker away from the pink trail.
- Recolor that one hit pixel to the background color.

Key insight
- The pink trail marks where the red marker came from, so the edit is found
  by ray-casting from the red marker in the opposite direction.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="c7f57c3e" lines="22-35">
<description>
Swap every sprite to the opposite prototype while preserving scale and position.

Input structure:
- bg = most common color. Two distinct small sprite patterns (prototypes) appear at various scales.
- Each sprite instance is a scaled copy of one of the two prototypes, placed on the grid.

Rule:
- Identify the two prototype patterns from the smallest-area sprites.
- For each sprite, determine which prototype it matches and its integer scale factor.
- Replace it with the other prototype at the same scale, aligned by an internal anchor region (color-1 pixels).

Key insight: Every sprite toggles to the alternate prototype; the anchor sub-region inside each sprite pins the alignment so swapped sprites land in the correct position regardless of scale.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="996ec1f3" lines="16-31">
<description>
Compress the noisy four-region board split by a solid cross into a 3x3 color map.

Input structure:
- A single full row and a single full column have the same solid separator color.
- The separator cross divides the input into four nonempty rectangular regions.
- Each region has a unique dominant color, with other colors acting as noise.

Rule:
- Find the unique solid same-color row/column pair.
- Fill the middle row and middle column of the 3x3 output with the separator color.
- Put the unique dominant color of each quadrant into the corresponding corner.

Key insight:
- The input grid is a noisy enlarged version of the desired 3x3 layout; the cross marks the coarse cell boundaries.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="ac0c2ac3" lines="7-19">
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

<reference_puzzle index="13" source="train" puzzle_id="5e6bbc0b" lines="42-56">
<description>
Compact the checkerboard's moving color toward the edge marked by the teal pixel, leaving a maroon swept trail on the marker line.

Input structure
- A blue/black checkerboard with exactly one teal marker replacing a blue pixel on a non-corner edge.
- The marker's checkerboard parity identifies the moving color (blue); the opposite parity is background (black).

Rule
- If the marker is on the left or right edge, compact moving-color pixels in every row toward that side.
- If the marker is on the top or bottom edge, compact moving-color pixels in every column toward that side.
- Preserve the teal marker at its edge cell.
- On the marker's row or column, color cells swept by shifted moving pixels maroon unless the final compacted line occupies them.

Key insight
The marker selects both the compression direction and the checkerboard color class that moves; maroon records the movement trail only along that marked line.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="dfadab01" lines="79-93">
<description>
Expand each colored control pixel into its corresponding fixed 4x4 glyph.

Input structure:
- Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
- Some inputs also show a sample 4x4 glyph in the output color, with a same-rule marker just below-right of the sample.

Rule:
- Replace each marker color by its learned 4x4 glyph: 2->yellow frame, 3->blue ring, 5->pink diagonal blocks, 8->orange X.
- Paint each glyph with its top-left corner at the marker pixel, clipping at the grid boundary.
- Remove the sample glyphs and skip their below-right label markers.

Key insight:
- The colored pixels are control pixels for a fixed glyph alphabet; non-control sample glyphs are only legend entries, not preserved scene objects.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="41ace6b5" lines="7-26">
<description>
Normalize vertical two-color bars between paired marker columns around the marker rows.

Input structure
- The background is the dominant color.
- A row of red marker pixels sits directly above a row of gray marker pixels
  in the same alternating columns.
- The columns between adjacent markers contain vertical bars made from a
  contiguous teal run above a contiguous blue run.

Rule
- Keep the marker pixels and all non-bar background cells fixed.
- Let h be the maximum teal-run height among the bar columns; paint teal in
  every bar column for h rows ending at the red marker row.
- In each bar column, move its blue run to start at the gray marker row and
  fill all lower cells in that column with maroon.

Key insight
- The marker rows are a baseline: teal is normalized into a common upper
  band, while each blue run keeps its own height below the baseline.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="9f41bd9c" lines="10-30">
<description>
Move the side-attached foreground object to the opposite wall, shear its lower rows inward, and paint the crossed floor cells.

Input structure:
- The top row is the upper background color and the bottom row is the
  floor color.
- A single non-background, non-floor object touches exactly one vertical
  grid edge and rests immediately above the first solid floor row.

Rule:
- Clear the source object to the upper background color.
- Repaint each object row on the opposite side: the first three relative
  rows keep the edge alignment, and each lower row shifts one more pixel
  inward from that side.
- On the first floor row, repaint the traversed side-to-object segment
  maroon, stopping just before the final outer bottom support pixel.

Key insight:
- The side of the source object gives the horizontal direction, while the
  row index inside the object gives the one-cell-per-row diagonal shear.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="2faf500b" lines="30-44">
<description>
Remove each fracture marker line and push the surrounding object pixels one cell away from it.

Input structure
- A background grid with 8-connected foreground objects made from a body color and a rarer marker color.
- Each object has a non-square bbox; the marker pixels form one complete fracture line across the shorter bbox axis.

Rule
- Erase all marker pixels.
- For a wide object, each row has one marker; body pixels left of it shift one cell left and body pixels right of it shift one cell right.
- For a tall object, each column has one marker; body pixels above it shift one cell up and body pixels below it shift one cell down.

Key insight
- The marker line is a crack through the object, so the two sides separate by one pixel perpendicular to the crack.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="136b0064" lines="25-39">
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

<reference_puzzle index="19" source="train" puzzle_id="25c199f5" lines="30-44">
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

<reference_puzzle index="20" source="eval" puzzle_id="cb2d8a2c" lines="13-26">
<description>
Extend a green path from one grid edge to the opposite, routing around red bars at a distance dictated by their blue pixel count.

Input structure:
- bg = azure (8). Red/blue bars (color 2/1) act as obstacles.
- A single green (3) pixel sits on a grid border — the path start.

Rule:
- Each bar's keep-out radius equals its number of blue pixels; the path must stay farther than that from every bar cell.
- The green pixel extends inward toward the opposite edge, turning at right angles to navigate around forbidden zones.
- After the path is drawn, all blue pixels become red.

Key insight: Blue pixels on a bar encode its danger radius — more blue means a wider no-go zone the path must respect.
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
11. 可以少出现以下三种类型的创新：查表填充、计数→索引、箭头→方向+距离；多些其他创新组合。

生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。
