你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="2d0172a1" lines="209-224">
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

<reference_puzzle index="2" source="train" puzzle_id="98c475bf" lines="62-76">
<description>
Erase the complete interior glyph and expand each endpoint marker pair into its color's full glyph.

Input structure
- The grid is 20x20 with background inside two same-color full-height vertical frame bars.
- Inside the frame there is one complete color glyph and one or more other colors appearing only as two endpoint markers in the same row.
- Marker endpoints are always in the first and last interior columns.

Rule
- Keep the vertical frame unchanged.
- Erase any complete interior glyph.
- For each endpoint marker pair, use that marker color's learned glyph template and anchor its horizontal bar on the marker row.

Key insight
- Color identifies which glyph shape to draw; the two endpoint pixels are not copied objects, they are anchors for reconstructing that color's full symbol.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="2b9ef948" lines="51-65">
<description>
Move the yellow 3x3 frame by the displacement encoded by the gray path, then draw its diagonal yellow corner rays.

Input structure:
- The background is black.
- One 3x3 yellow frame has a non-yellow control-color center.
- A second control-color pixel is connected through gray path pixels to a lone yellow endpoint.

Rule:
- Use the vector from the second control-color pixel to the lone yellow endpoint as the frame's translation.
- Fill the whole output with the control color.
- Draw the translated yellow frame and extend yellow diagonal rays outward from its four corners to the grid edge.

Key insight: The gray path's turns are irrelevant; its two colored endpoints encode the net displacement to apply to the frame.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="cbebaa4b" lines="51-65">
<description>
Assemble scattered jigsaw sprites around a central yellow anchor by pairing red connector pixels.

Input structure:
- bg = black (0). Several colored sprites scattered on the grid.
- One anchor sprite contains a 3×3 yellow (4) block.
- Each sprite has red (2) connector pixels along its edges.

Rule:
- The anchor stays in place; remaining sprites attach one at a time.
- Each sprite is translated so that one of its red pixels overlaps an unpaired red pixel on the growing assembly.
- Every red connector must pair with exactly one other, forming a complete joined structure.

Key insight: Red pixels act as jigsaw pegs — each must match exactly one counterpart, and the assembly order is determined by which unpaired connectors remain.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="689c358e" lines="72-88">
<description>
Mark each arrow-like sprite's pointed frame endpoint with its color and the opposite endpoint black.

Input structure:
- The grid has a one-pixel solid rectangular frame around a dominant interior background.
- Inside the frame are monochrome six-pixel sprites shaped as asymmetric T arrows.
- Each sprite has a length-4 bar and a length-3 crossbar one cell closer to the pointed end.

Rule:
- For each sprite, follow its bar axis from the shorter end to the frame.
- Paint that frame endpoint with the sprite color.
- Paint the opposite endpoint on the same row or column black.
- Leave the interior background, sprites, and all other frame pixels unchanged.

Key insight:
- The direction is encoded by the crossbar's off-center position: the nearer end is the pointing end.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="c803e39c" lines="31-40">
<description>
Expand the second mask into a block grid by stamping the first mask with the two payload colors.

Input structure
- A background grid has four panels separated by three full-height separator columns.
- The first two panels contain same-size binary masks; the last two contain solid same-size color squares.
Rule
- Use the second mask as an outer layout. For each foreground cell in that outer mask, place a tile equal to the first mask, coloring its foreground with the first solid-square color and its background with the second solid-square color; outer background cells become all second solid-square color.
Key insight
- The output is the Kronecker product of the two binary masks, with the two right-hand panel colors providing the palette.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="c4d1a9ae" lines="28-48">
<description>
Fill each vertical foreground band's background cells with the color of the next band to its right.

Input structure
- The dominant color is the background.
- Each other color occupies a separate vertical band defined by that color's
  minimum and maximum columns.
- Within a band, all cells are either that band's color or background.

Rule
- Order the bands from left to right.
- In each band, replace background pixels with the color of the next band in
  that order, wrapping from the rightmost band back to the leftmost.
- Preserve all existing foreground pixels and all background pixels outside
  the bands.

Key insight
- The column bands form a cyclic palette legend: a band's gaps are completed
  by borrowing the color from its right-hand neighbor, not by changing the
  band's existing pixels.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="d8e07eb2" lines="29-47">
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

<reference_puzzle index="9" source="train" puzzle_id="41ace6b5" lines="7-26">
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

<reference_puzzle index="10" source="eval" puzzle_id="16b78196" lines="6-20">
<description>
Assemble colored sprites into a central bar's notches, chaining successive pieces along fixed-width strips.

Input structure:
- bg = most common color. Largest monochrome component is a solid "bar" with boundary notches (bg holes opening to its bbox edge).
- Several smaller colored sprites scattered around the bar.

Rule:
- Each bar notch is filled by a root piece whose shape exactly covers the hole.
- The root defines a strip (column-span or row-span) for its chain.
- Successive pieces fill the bg cells inside the previous piece's bbox, locked to the same strip width.
- Multiple chains grow independently from different bar notches.

Key insight: The root piece's extent perpendicular to the chain direction locks the strip width for all subsequent pieces in that chain.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="5a719d11" lines="44-59">
<description>
Within each horizontal pair of two-color panels, swap the motif masks and recolor each pasted mask with the source panel's base color.

Input structure:
- A separator color forms full blank rows and at least one full blank column.
- The non-separator area is split into row bands, each containing exactly two same-shaped rectangular panels.
- Each panel has a dominant base color and one minority motif color.

Rule:
- For every row band, clear each panel to its own base color.
- Paste the right panel's motif mask into the left panel using the right base color.
- Paste the left panel's motif mask into the right panel using the left base color.

Key insight:
- The motif colors themselves are distractors; only each motif's shape mask and the two panel base colors are exchanged.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="4c7dc4dd" lines="7-21">
<description>
Learn a transformation rule from a training pair of framed sprites, then apply it to a test sprite.

Input structure:
- Grid contains several colored square frames (hollow borders of a single color).
- Each frame's interior holds a small pattern sprite; one frame's interior is blank (all black).
- Two frames form a training pair (their interiors share the same color palette); one remaining frame is the test case.

Rule:
- Identify the training pair by matching interior color sets; the other non-blank frame is the test input.
- Infer the rule mapping source→destination (e.g. reflection, color inversion, gap filling, endpoint extraction, or checkerboard complement).
- Apply that same rule to the test sprite; output the transformed interior.

Key insight: The puzzle is an analogy — "A is to B as C is to ?" — where you must discover the visual rule from one example pair and generalize it.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="62593bfd" lines="32-46">
<description>
Sort objects to top or bottom edge by whether their shape (up to rotation/flip) is shared.

Input structure:
- bg = most common color.
- Several multicolor connected components (8-connected) scattered across the grid.

Rule:
- Compute each object's canonical shape (binary mask, invariant to rotation and flip).
- Objects whose shape appears in two or more components → translate to TOP edge (row 0).
- Objects whose shape is unique → translate to BOTTOM edge (row H−h).
- Horizontal position (left column) is preserved; only vertical translation changes.

Key insight: The sorting criterion is shape frequency (rotation-invariant), not area — duplicated shapes go up, unique shapes go down.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="c3fa4749" lines="79-96">
<description>
Recolor each side-marked multicolor object embedded in a monochrome panel to its marker color.

Input structure:
- Same-color 4-connected components form monochrome panel regions.
- The bounding box of a panel may contain 4-connected non-panel-color objects.
- A target object touches exactly one side of its panel bbox, and all of its
  pixels on that side have one marker color.

Rule:
- For every target object, fill any panel-colored holes enclosed by the object.
- Recolor the object and filled holes to the side marker color.
- Leave all other pixels unchanged.

Key insight:
- A multicolor object's single contact side identifies both which object should
  be normalized and which color should replace it.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="df9fd884" lines="33-47">
<description>
Move the colored object from the lower bracket lane to the upper bracket lane and drop it to the bottom edge.

Input structure
- The background is the dominant color.
- One frame color forms exactly two 4-connected bracket objects: a top-edge bracket open downward and a bottom-edge bracket open upward.
- One other 4-connected foreground object sits in the lower bracket's interior column span.

Rule
- Remove the movable object from its original cells.
- Keep its shape and color unchanged.
- Paste it with its bbox columns aligned to the top bracket's interior columns and with its bbox bottom aligned to the grid bottom.

Key insight
- The lower bracket identifies the object's source lane; the upper bracket provides the target lane, and the final vertical placement is bottom-aligned.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="446ef5d2" lines="8-20">
<description>
Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

Input structure:
- bg = most common color. Non-bg rectangular pieces scattered on the grid.
- 3-pixel L-marker (2x2 minus one cell) tags the corner; the gap is the corner cell.

Rule:
- rect_col = most common color across all piece pixels; all outer rectangle edges must be rect_col.
- Translate pieces (no rotation) to tile the rectangle; touching edges must match, with rect_col as wildcard.

Key insight: rect_col acts as wildcard during edge matching — only non-rect_col pixels at shared boundaries must agree.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="de809cff" lines="86-103">
<description>
Turn black dents in two colored regions into marked opposite-color patches.

Input structure:
- Black background with two large foreground colors.
- The large regions contain small black dents and a few one-cell color specks
  or protrusions along region boundaries.

Rule:
- Locally clean the two foreground regions: isolated specks become black, and
  one-cell protrusions surrounded by the other region take that region's color.
- Each original black dent whose cleaned 3x3 neighborhood is mostly one
  foreground color becomes the center of a 3x3 patch in the opposite color.
- The dent center itself is recolored teal (8).

Key insight: The teal pixels mark the original black defects; the surrounding
3x3 patch uses the other foreground color, after first removing boundary noise.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="2a28add5" lines="8-22">
<description>
Paint each pink control pixel as a teal horizontal bar whose reach equals the foreground counts on that row.

Input structure
- The grid has a dominant background color and one or more pink control pixels.
- Each marker row has exactly one pink pixel; all other non-background pixels
  in that row are counters.
Rule
- Start from an all-background output grid.
- For each pink pixel, count non-background pixels strictly to its left and
  strictly to its right in the same row.
- Paint teal from c - left_count through c + right_count on that row.
Key insight
- The foreground pixels are not copied; only their counts on each side of
  each pink marker encode the teal bar length.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="7acdf6d3" lines="25-42">
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

<reference_puzzle index="20" source="train" puzzle_id="ecb67b6d" lines="138-155">
<description>
Recolor the grey diagonal-chain regions in an orange field to teal.

Input structure
The input contains only grey pixels and orange pixels.  Grey pixels form
several 8-connected regions embedded in the orange field.

Rule
Decompose each grey 8-connected region into 4-connected lobes.  Recolor the
whole region teal only when it contains a straight diagonal run of at least
three pixels and those lobes meet only by diagonal corner contacts in a
non-branching diagonal chain or cycle; a single 4-connected region is
recolored only when it is a square diagonal staircase.

Key insight
The teal regions are not selected by size or position; they are the grey
regions whose internal connectivity depends on diagonal corner joins, while
same-edge cracks, rectangles, bars, and branching/noisy regions remain grey.
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
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
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

