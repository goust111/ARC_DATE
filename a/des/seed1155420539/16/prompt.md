你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="6e4f6532" lines="26-41">
<description>
Dock train sprites onto station groups by matching coupler count, then orient via divider-color markers.

Input structure:
- bg; colored stripes (edge + internal) partition grid into rectangular regions.
- Color 8 forms train bodies; color 9 marks both isolated stations and train-attached couplers.
- Stripe-colored pixels on trains are orientation markers.

Rule:
- Stations = 8-connected 9-clusters not adjacent to any 8. Trains = components containing 8.
- Match train to station group where |couplers| == |station size|.
- Rotate/flip train so marker pixels face the same-color boundary stripe of the target region.
- Translate so couplers align exactly onto station cells; erase original position.

Key insight: Coupler count selects the docking target; marker color encodes which boundary stripe to face, fixing the unique orientation.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="ba1aa698" lines="41-55">
<description>
Extrapolate the moving sprite from a horizontal sequence of shared-border frames into the next standalone frame.

Input structure
- The grid is a row of equal-width framed panels sharing their vertical border columns.
- Each panel has the same fill color and one identical interior sprite at the same local columns.
- The sprite's top row changes by a constant amount from panel to panel.

Rule
- Build one output panel with the same height, frame color, fill color, and sprite.
- Continue the sprite's diagonal motion one more panel to the right.
- Because the output is a standalone panel, advance by the full panel width, not the shared-border spacing.

Key insight
- The panels are consecutive frames of motion; the output is the next frame after the sequence.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="ecb67b6d" lines="138-155">
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

<reference_puzzle index="4" source="eval" puzzle_id="cb2d8a2c" lines="13-26">
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

<reference_puzzle index="5" source="train" puzzle_id="46c35fc7" lines="21-34">
<description>
Rotate the colors around every 3x3 foreground ring, with corners and side-centers cycling in opposite directions.

Input structure
- The grid has a dominant background color.
- Each 4-connected foreground object is a complete 3x3 ring: eight foreground pixels around a background center.

Rule
- For each ring, move corner colors one step around the four corners.
- Move side-center colors one step around the four side centers in the opposite direction.
- Leave the background center and all other background pixels unchanged.

Key insight
- Corners and side centers form two separate four-cycles on the same 3x3 ring.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="dfadab01" lines="79-93">
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

<reference_puzzle index="7" source="train" puzzle_id="5ecac7f7" lines="23-36">
<description>
Combine the left, middle, and right ordered objects from three separated panels.

Input structure
- The grid is three HxH panels separated by one-cell vertical separator bars.
- The background is the dominant color, and each panel contains exactly three same-color 4-connected foreground objects.

Rule
- Build a new HxH background canvas.
- From the left panel copy its leftmost object, from the middle panel copy its middle object, and from the right panel copy its rightmost object.

Key insight
- The separator bars create three aligned layers, and the panel's own left-to-right object order selects which object survives in the merged output.
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

<reference_puzzle index="9" source="eval" puzzle_id="332f06d7" lines="9-23">
<description>
Slide a solid square through a winding channel to reach a same-sized target square.

Input structure:
- bg (most common color). A river color forms a winding corridor.
- Two same-sized NxN solid squares on the river: mover and destination, each a distinct color.

Rule:
- The mover square travels along the river toward the destination.
- It can only pass through positions where the full NxN area is river.
- If reachable, the mover replaces the destination; otherwise it stops at the farthest reachable point.
- The mover's original position is filled with the river color.

Key insight: Chokepoints narrower than N pixels block the square, forcing it to stop short.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="fc10701f" lines="20-35">
<description>
Move the orange rectangle onto the maroon rectangle and mark gated corridor crossings in red.

Input structure
- The dominant background contains black obstacle segments.
- There is one solid orange rectangle and one solid maroon rectangle of the same size.
- The two rectangles share either their row span or their column span, so the move is straight and axis-aligned.

Rule
- Erase the original orange rectangle to the background.
- Replace the maroon rectangle with the orange rectangle.
- Between the two rectangles, paint red across the moving rectangle's width or height only at corridor cells flanked by black obstacle segments perpendicular to the movement.

Key insight
- The black segments define gates in the straight corridor; only those gated openings, not every swept background cell, become red.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="88e364bc" lines="102-115">
<description>
Slide yellow pixels inside walled regions in the direction dictated by border-legend arrows.

Input structure:
- bg = most common color; yellow pixels sit inside enclosed regions bounded by colored walls.
- Border legends: small rectangles on the grid edge whose fill color C matches a region's border color. Each legend contains a center dot and an indicator dot encoding a direction.

Rule:
- For each enclosed region, identify its border color C and look up the legend with fill color C.
- The direction from the legend's center dot to its indicator dot gives the slide direction.
- All yellow pixels in that region slide in that direction, stacking against the far wall.

Key insight: Legend fill color links to region border color; the center→indicator offset encodes the movement vector.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="689c358e" lines="72-88">
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

<reference_puzzle index="13" source="train" puzzle_id="412b6263" lines="39-54">
<description>
Infer the background, rotate the whole input 90 degrees counterclockwise, and stack two framed copies vertically.

Input structure:
- Blue is absent from the input.
- One rare marker color appears in exactly two pixels.
- The background is the unique non-marker color whose removal leaves exactly two 4-connected foreground objects, each with one marker pixel and one body color.
- The whole grid, including background and foreground objects, is the sprite to transform.

Rule:
- Rotate the entire input grid 90 degrees counterclockwise.
- Create a blue one-pixel frame around two vertical copies of the rotated grid.
- Use the input background color for the two endpoints of every horizontal frame/separator row.

Key insight: The output is not a crop of the foreground objects; it is two copies of the complete rotated input, separated and enclosed by a newly added blue border.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="18286ef8" lines="17-31">
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

<reference_puzzle index="15" source="train" puzzle_id="342dd610" lines="18-36">
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

<reference_puzzle index="16" source="eval" puzzle_id="97d7923e" lines="81-94">
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

<reference_puzzle index="17" source="train" puzzle_id="37ce87bb" lines="33-47">
<description>
Compute a grey result bar by adding teal bar heights and subtracting red bar heights.

Input structure
- The grid has a dominant background and several bottom-aligned one-cell-wide vertical bars.
- Teal bars are positive operands; red bars are negative operands.
- The penultimate column is an empty result slot.

Rule
- Keep every input bar unchanged.
- Compute result height = total teal bar height - total red bar height.
- Paint that many bottom cells of the penultimate column grey.

Key insight
- The bars encode arithmetic by height, with color providing the sign.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="67e490f4" lines="8-22">
<description>
Fill holes inside a large template frame using shape-matched outside sprites.

Input structure:
- bg = most common color. One large monochrome frame object (template).
- Inside the frame: holes of various shapes (non-frame-color regions).
- Outside the frame: small monochrome sprites scattered on bg.

Rule:
- Crop to the frame's bounding box; fill entirely with frame color.
- For each hole, find an outside sprite whose shape matches (rotation/flip allowed).
- Fill the hole with the matched sprite's color. If multiple sprites match, use the most frequent color.

Key insight: Sprites act as shape-coded color keys — their shape identifies which hole they fill, their color determines the fill color.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="b7256dcd" lines="18-32">
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

<reference_puzzle index="20" source="train" puzzle_id="db615bd4" lines="111-130">
<description>
Thicken the sparse frame and stamp the external sparse sprites into it in positional order.

Input structure
The grid has a dominant background color, a regular lattice color, one large
sparse rectangular frame, and one or more external monochrome sprites whose
pixels also sit on a two-step lattice.

Rule
Use the largest sparse rectangular perimeter as the frame. Clear that frame
region and the external sprite pixels to background, draw a solid one-pixel
border in the frame color, solidify each sparse sprite into its contiguous
two-step mask, and place the sprites inside the frame with equal gaps.
Wide frames receive sprites left-to-right by source position; tall frames
receive them top-to-bottom by source position.

Key insight
The frame supplies ordered slots: the sprite-to-slot assignment is by visible
source order, not by numeric color or by matching individual cells.
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

