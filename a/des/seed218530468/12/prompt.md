你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="8a6d367c" lines="52-67">
<description>
Complete the framed area by expanding the unique upper template that contains the existing seed pixels.

Input structure:
- A teal background contains small colored template masks above a larger rectangular frame.
- The frame encloses one fill color arranged as a partial block-scaled copy of one template.
- The frame interior dimensions are an integer square scale of the matching template bbox.

Rule:
- Crop to the rectangular frame.
- Identify the unique upper template whose block-scaled mask covers every existing fill pixel inside the frame.
- Reset the frame interior to background and paint the whole scaled mask with the fill color.

Key insight:
The lower colored pixels are seed cells that reveal which upper template should be completed inside the frame.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="cf5fd0ad" lines="10-22">
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

<reference_puzzle index="3" source="eval" puzzle_id="a6f40cea" lines="74-88">
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

<reference_puzzle index="4" source="train" puzzle_id="2ccd9fef" lines="173-187">
<description>
Predict the next panel in a framed sequence by continuing each scheduled colored interval.

Input structure
- The grid is made from three or more equal panels in one row or one column.
- The final input panel is the static target template; every earlier panel differs from it only by dynamic colors absent from the template.
- Each dynamic color appears in a regular temporal schedule across the earlier panels.

Rule
- Keep the final template panel as the output canvas.
- For every dynamic color whose schedule reaches the next panel, compare its last two appearances.
- Extrapolate each one-dimensional interval along the color's changing row or column axis and paint that predicted mask.

Key insight
- The task is not asking for the last visible panel, but for the next frame after the blank/template slot.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="ecb67b6d" lines="138-155">
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

<reference_puzzle index="6" source="train" puzzle_id="48634b99" lines="14-27">
<description>
Advance the marker from one half of a vertical bar to the same half of the next longer bar.

Input structure
- The grid has one dominant background color and exactly two foreground colors.
- One foreground color forms many vertical one-cell-wide bars; the other appears as a single vertical marker segment.
- The marker segment occupies exactly the top or bottom half of its containing vertical bar.
Rule
- Recolor the current marker segment back to the bar color.
- Find the unique all-bar-color vertical bar whose height is two pixels greater than the marked source bar.
- Recolor the same half of that next longer bar to the marker color.
Key insight
- Each bar's total height encodes an ordered size, and the marker advances from a 2k bar to the corresponding 2(k+1) bar while preserving whether it marks the top or bottom half.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="fd02da9e" lines="26-41">
<description>
Replace each colored corner marker with its corner-specific inward stencil in the same color.

Input structure
- The grid is 8x8 with a single dominant background color.
- Foreground pixels are marker pixels, each located at one of the four corners.
- A marker's color is carried to the stencil selected by its corner position.

Rule
- Erase each corner marker back to the background.
- Top corner markers become an inward 2x2 block one cell away from the top and side edges.
- Bottom corner markers become the mirrored three-pixel inward bent stencil shown by the bottom-corner examples.

Key insight
- The marker position selects the fixed stencil; the marker color only supplies the paint color, so multiple corners apply independently.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="dd6b8c4b" lines="9-22">
<description>
Move the closest reachable maroon pixels into the 3×3 block around the unique red center.

Input structure:
- bg = orange floor, magenta walls, scattered maroon (9) pixels.
- A unique red (2) pixel marks the center of a 3×3 destination block.

Rule:
- Maroon pixels connected to the destination through floor/maroon cells (not blocked by walls) are candidates.
- Rank candidates by distance to the destination; the closest ones fill the block in row-major order.
- Moved maroons are erased (replaced with floor color).

Key insight: Wall barriers restrict connectivity — only maroon pixels reachable via flood-fill through the floor qualify, so proximity alone isn't enough.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="4a1cacc2" lines="8-23">
<description>
Fill the rectangle from the lone foreground pixel to its nearest grid corner.

Input structure
- The grid is a uniform background field with exactly one foreground pixel.
- The foreground pixel's color is the color used for all added pixels.

Rule
- Find the grid corner with the smallest Manhattan distance to the foreground pixel.
- Recolor every pixel in the inclusive axis-aligned rectangle bounded by that
  corner and the foreground pixel to the foreground color.

Key insight
- The single pixel is one corner of the rectangle; the opposite corner is the
  nearest corner of the whole canvas.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="9f669b64" lines="26-48">
<description>
Push the central solid rectangle to the grid edge, splitting the perpendicular target rectangle around its path.

Input structure
- The background is the dominant color.
- There are exactly three monochrome foreground objects arranged on one
  horizontal or vertical axis.
- The middle object on that axis is a solid rectangular moving object.
- One end object is a solid rectangle or bar whose long side is perpendicular
  to the movement axis; this is the target to split.

Rule
- Move the middle rectangle along the axis toward the target until it is
  flush with the grid edge in that direction.
- Clear the original middle rectangle and the original target.
- Split the target through the moving rectangle's two-cell-wide path and
  shift the two halves one cell outward perpendicular to the movement.
- Keep the other end object unchanged.

Key insight
- The non-target end object acts as a pusher; the central rectangle travels
  away from it through the perpendicular target, opening a matching gap.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="ac6f9922" lines="18-40">
<description>
Summarize the enclosed table slots by the single object color inside each slot, using the slot background color for empty slots.

Input structure
- The most common color forms the outer background and all enclosed slot
  backgrounds.
- The enclosed background components form a complete rectangular array of
  slots, ordered by their positions.
- Each slot contains either no other color or exactly one object color.

Rule
- Find all 4-connected components of the most common color that do not touch
  the input edge.
- Sort their bounding boxes into row and column order to define the output
  table.
- For each slot, output its unique non-background color if present; otherwise
  output the background color.

Key insight
- The large grid is an expanded table: the enclosed background components are
  the cells, and any colored object drawn inside a cell is that cell's summary
  value.
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

<reference_puzzle index="13" source="train" puzzle_id="9841fdad" lines="15-39">
<description>
Expand the colored runs from the left reference panel into the blank right panel.

Input structure:
- A uniform rectangular frame surrounds the grid.
- One full-height divider of the frame color splits a left reference panel
  from a blank right panel.
- Both panels have one background margin column on each side of their active
  columns; the left active columns contain two-column endpoint slots, possibly
  connected by same-color horizontal runs.

Rule:
- Keep the original grid and paint only in the right panel's active columns.
- A source run exactly matching the left endpoint slot is copied to the left
  two columns of the right active span.
- A source run exactly matching the right endpoint slot is copied to the right
  two columns of the right active span.
- A source run connecting the endpoint slots is stretched across the entire
  right active span.

Key insight:
The left panel is a compressed legend for the right panel: endpoint markers
stay two pixels wide, while connectors between endpoints expand to fill the
distance between them.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="2b9ef948" lines="51-65">
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

<reference_puzzle index="15" source="train" puzzle_id="396d80d7" lines="18-30">
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

<reference_puzzle index="16" source="train" puzzle_id="2601afb7" lines="30-45">
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

<reference_puzzle index="17" source="train" puzzle_id="98c475bf" lines="62-76">
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

<reference_puzzle index="18" source="train" puzzle_id="d255d7a7" lines="99-114">
<description>
Move the patterned edge band to the opposite edge along every row or column whose band slice contains the trail color.

Input structure
- The dominant color is the background.
- The two foreground colors are a frequent trail color and a rarer marker color.
- All marker pixels lie nearest to one grid edge, and that edge contains trail-color runs that define the source-band thickness.

Rule
- Choose the source edge identified by the marker pixels.
- A row or column perpendicular to that edge moves if its source-band slice contains the trail color.
- Copy that slice unchanged to the opposite edge, erase the remaining trail cells to background, and keep marker-only lines fixed.

Key insight
- The marker color identifies the active edge, while the trail color identifies which parallel rows or columns carry an edge-band slice across the grid.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="8dab14c2" lines="120-133">
<description>
Complete each boundary tab or notch by adding the complementary feature on the opposite side of the foreground object.

Input structure
- The grid has exactly two colors: a dominant background and one foreground object.
- The foreground object is a thick rectilinear shape with one-pixel tabs and notches on its boundary.

Rule
- A foreground tab is a one-neighbor boundary pixel; project through its full row or column span and remove the opposite edge pixel to make a matching notch.
- A background notch is a one-pixel cut with three foreground 4-neighbors; project across the full row or column span and add a tab just outside the opposite edge.
- Notches immediately beside a tab source are not independent sources, because that gap belongs to the tab's local boundary shape.

Key insight
- The changed pixels are not color-specific noise; they are paired jigsaw-like boundary features on opposite sides of the same thick row or column cross-section.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="8fff9e47" lines="8-25">
<description>
Turn a 4-cell-wide strip of 2x2 blocks into four nested corner-fill patterns.

Input structure
The input is an even-sized rectangle whose shorter dimension is four pixels.
It can be partitioned exactly into 2x2 blocks, giving one block for each layer
of the output quadrants.

Rule
Read the 2x2 blocks in reverse row-major order: bottom-right block first,
moving left, then moving upward. For each block, send its top-left,
top-right, bottom-left, and bottom-right pixels to the layer sequences of the
upper-left, upper-right, lower-left, and lower-right output quadrants.

Key insight
Each input 2x2 block supplies the four colors for one concentric layer, and
the layer order is the block order from bottom-right back to top-left.
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

