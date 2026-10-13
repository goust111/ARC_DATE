你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="c920a713" lines="87-102">
<description>
Untangle overlapping colored rectangular frames into concentric square frames ordered by their visible crossings.

Input structure
- The background contains several one-cell-thick axis-aligned rectangular frames, one color per frame.
- Frames overlap, and the visible color at an overlap records a directed crossing from the hidden frame to the visible frame.
- Each foreground color still has visible pixels on the perimeter of its full rectangular bounding box.

Rule
- Infer each color's full rectangular frame from the bounding box of its visible pixels.
- Add a directed edge A -> B when B is visible at a crossing of A and B's frame perimeters.
- Find the unique path that visits every frame color once, then render those colors as one-pixel-thick concentric square frames from outside to inside.

Key insight
- The output is not a crop of the input; the crossing graph gives a single frame order, and that order is redrawn as nested rings.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="fc10701f" lines="20-35">
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

<reference_puzzle index="3" source="train" puzzle_id="b1986d4b" lines="19-37">
<description>
Render the filled-square object counts as compact nested stacks ordered by count rank.

Input structure
A large noisy grid has a dominant background, scattered non-square noise, and
filled monochrome square objects. Each square side length has exactly one
object color, and side lengths form consecutive sizes.

Rule
Ignore non-square components. If the lowest square in the scene is of the
smallest side length, treat that one smallest square as a distractor and do
not count it. For count rank k from left to right, create one block from all
square sizes with at least k remaining objects; draw larger squares first and
smaller squares over their upper-left corner, then append one background
separator column.

Key insight
The output is a count histogram of square side lengths, but each histogram
slot is rendered as nested upper-left square masks using the side colors.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="db615bd4" lines="111-130">
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

<reference_puzzle index="5" source="train" puzzle_id="b7955b3c" lines="386-403">
<description>
Replace gray plus-shaped damage masks with the hidden colors of the rectilinear mosaic.

Input structure:
- Color 8 appears only as isolated 3x3 plus masks.
- The non-mask pixels are visible parts of a clean grid made from horizontal
  and vertical color regions.

Rule:
- Treat every color-8 pixel as unknown damage and leave all other pixels fixed.
- First copy unknown pixels from exact matching parallel rows or columns.
- Then fill any remaining mask cells from opposite-side continuations and the
  better-supported local straight edge.

Key insight:
- The gray pluses are occluders, not objects; the underlying colors are
  determined by repeated line templates and nearby axis-aligned boundaries.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="ba1aa698" lines="41-55">
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

<reference_puzzle index="7" source="train" puzzle_id="ff2825db" lines="18-36">
<description>
Select the majority interior marker color, then redraw the framed board as that color's frame plus its bounding-box outline.

Input structure:
- The top row is a copied palette made from adjacent equal color pairs.
- Below it is a one-cell-thick rectangular frame around a black interior.
- Non-black, non-frame interior pixels are marker colors; one marker color has
  the unique highest count.

Rule:
- Keep the palette row unchanged.
- Pick the uniquely most frequent interior marker color.
- Clear the lower board to black, recolor its outer frame with the picked color,
  and draw the picked color's interior bounding-box border.

Key insight:
- The noisy marker cloud is not preserved as pixels; only the majority marker
  color and its spatial extent define the output.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="35ab12c3" lines="73-88">
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

<reference_puzzle index="9" source="train" puzzle_id="abbfd121" lines="56-74">
<description>
Recover the repeating background pattern hidden under the largest solid rectangle.

Input structure:
- The grid is covered by a small row/column periodic pattern.
- Several non-pattern colors form solid rectangular foreground objects that
  overwrite the pattern.
- The requested output region is the largest of those solid rectangles.

Rule:
- Find every 4-connected monochromatic object whose bounding box is a filled
  rectangle with both dimensions greater than one.
- Treat those rectangles as masks, infer the smallest repeating tile that is
  consistent with all remaining visible pixels, and repaint the largest
  rectangle's coordinates with that tile.

Key insight: The rectangle colors are occluders; the visible pixels outside
them are enough to reconstruct the original periodic background underneath.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="b2bc3ffd" lines="8-23">
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

<reference_puzzle index="11" source="train" puzzle_id="fe45cba4" lines="72-89">
<description>
Move the left-edge fragment into its same-colored anchor fragment to complete a solid rectangle.

Input structure
- The grid has a dominant background color.
- Exactly one foreground color is split into two 4-connected objects: one
  touches the left edge, and the other is an anchor away from the left edge.
- The translated left-edge object and the anchor can form one filled rectangle.
Rule
- Find the unique split color whose left-edge object can be translated onto
  background cells beside the other same-colored object so their union is a
  solid axis-aligned rectangle.
- Clear the left-edge object's original cells to the background.
- Paint that object at the unique completing offset, preserving all other cells.
Key insight
- The active color is identified by object structure and rectangle completion,
  not by a fixed color id; the same-colored anchor object stays in place.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="3ad05f52" lines="128-142">
<description>
Paint the maze corridor and framed rooms lying on shortest paths between the two colored endpoint components.

Input structure
- The grid has a dominant background, one wall color forming thin orthogonal frames, and one endpoint/fill color.
- The endpoint/fill color forms exactly two 4-connected solid rectangular components; their larger side lengths define the framed room sizes.

Rule
- Treat wall pixels as blocked and every other pixel as passable.
- Find every passable pixel that lies on a shortest 4-connected path between the two endpoint components.
- Also fill any framed square room of an endpoint-derived size that intersects that shortest-path mask.

Key insight
- Shortest paths identify the route through open connector corridors, while the wall-bordered square rooms need to be expanded from the route centerline to their full interior.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="5ecac7f7" lines="23-36">
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

<reference_puzzle index="14" source="eval" puzzle_id="7b5033c1" lines="7-21">
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

<reference_puzzle index="15" source="train" puzzle_id="494ef9d7" lines="16-30">
<description>
Slide each blue/teal or yellow/orange row pair together while preserving its left-to-right order.

Input structure
- The grid has a black background.
- Each nonempty row contains exactly two distinct foreground pixels.
- Blue/teal and yellow/orange row pairs are the moving pairs; other color pairs are distractors.

Rule
- For each moving pair, keep the left pixel fixed and move the right pixel left until it touches the left pixel.
- Clear the moved pixel's original location, and leave non-moving rows unchanged.

Key insight
- Only same-hue-family pixel pairs attract horizontally; separated unrelated colors do not compact.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="28a6681f" lines="8-23">
<description>
Simulate water pixels falling under gravity into basins formed by colored obstacles.

Input structure:
- bg = 0. Water pixels = color 1 (blue).
- Other colors form static solid structures (walls, steps, containers).
- Water starts in unstable positions above or beside basins.

Rule:
- Each water pixel falls straight down through bg cells.
- When blocked below, it slides diagonally through the unique open side cell.
- When resting on a surface, it slides horizontally toward the nearest gap where it can fall again.
- Process repeats until no water pixel moves.

Key insight: Water seeks the deepest reachable basin via diagonal and horizontal sliding, not just vertical fall.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="22208ba4" lines="12-27">
<description>
Move the repeated boundary color's rectangles one block inward while leaving singleton boundary rectangles in place.

Input structure:
- The background is the most common color.
- Foreground objects are solid monochrome rectangles touching a grid corner.
- Exactly one foreground color appears as multiple separate 4-connected objects; other colors appear once.

Rule:
- Clear the multiple objects of the repeated color from their original positions.
- For each cleared rectangle, translate it away from the row and column border it touches by exactly its height and width.
- Keep singleton-color objects unchanged; overlapping translated rectangles of the same color merge.

Key insight:
- Multiplicity selects the moving color, and each boundary rectangle's touched edges determine its inward diagonal translation.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="2de01db2" lines="28-41">
<description>
Invert each row's native foreground color against the black background.

Input structure
- A black-background grid whose rows each have one native foreground color plus optional distractor colors.
- The native row color is the most frequent non-bg color in that row; if tied, its run starts leftmost among tied colors.

Rule
- For each row independently, turn pixels of the native row color into bg.
- Turn every other pixel in that row, including bg and distractor colors, into the native row color.

Key insight
- Distractor colors are not preserved objects; they mark cells that participate in the binary complement of the row's native color.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="46c35fc7" lines="21-34">
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

<reference_puzzle index="20" source="train" puzzle_id="50c07299" lines="8-21">
<description>
Move the diagonal foreground segment to the next longer run along the same up-right diagonal.

Input structure
The grid has one background color and one foreground color. The foreground pixels form a
contiguous one-pixel-wide segment on a single anti-diagonal.

Rule
Erase the current segment. If its columns are c0..c1 and its length is L, draw the same
color on columns c1+1..c1+L+1 along the same anti-diagonal.

Key insight
The segment advances past its current upper-right end by one whole segment position, and
the next segment is one pixel longer than the current one.
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

