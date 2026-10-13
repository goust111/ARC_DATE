你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="5587a8d0" lines="11-27">
<description>
Redraw foreground objects as nested square rings ordered by object size.

Input structure:
- The grid has one dominant background color.
- Every foreground object is a 4-connected monochromatic component.
- Foreground object areas are distinct.

Rule:
- Sort the foreground objects by decreasing pixel count.
- Create a square output with side length 2 * number_of_objects - 1.
- The largest object color fills the outer ring, each next object color fills
  the next inner ring, and the smallest object color fills the center pixel.

Key insight: The objects' original shapes do not get copied; their areas
determine only the outside-to-inside order of the nested square pattern.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="95755ff2" lines="45-60">
<description>
Paint straight rays from colored side markers into the hollow diamond frame until an interior foreground pixel blocks each ray.

Input structure
- The background contains one complete one-pixel-thick diamond frame.
- Colored marker pixels outside the diamond sit on the top, bottom, left, or right side of the frame's bounding box.
- Any non-background, non-frame pixels inside the diamond are blockers that remain unchanged.

Rule
- Each outside marker emits inward perpendicular to its side of the diamond bounding box.
- Along that row or column, paint only background pixels strictly inside the diamond with the marker color.
- Stop a ray when it reaches an existing interior foreground pixel; keep the frame and all original colored pixels unchanged.

Key insight
- The diamond frame defines a strict interior, while the side markers are column or row emitters whose color fills the visible interior interval until blocked.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="f8cc533f" lines="10-25">
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

<reference_puzzle index="4" source="train" puzzle_id="7e2bad24" lines="91-106">
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

<reference_puzzle index="5" source="train" puzzle_id="7ec998c9" lines="14-28">
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

<reference_puzzle index="6" source="eval" puzzle_id="a395ee82" lines="9-24">
<description>
Stamp translated copies of a monochrome template guided by a singleton pixel layout.

Input structure:
- bg = background. A large monochrome shape (template, color T).
- Singleton pixels elsewhere: exactly one is color T (layout origin); the rest share color O (overlap color).
- Singletons sit on a uniform-spacing lattice.

Rule:
- Convert each singleton's offset from the origin into coarse lattice units.
- Place a template copy at each offset, scaled by template height/width.
- The origin's copy is painted O instead of T.
- Any pixel covered by two or more copies is also recolored to O.

Key insight: The single T-colored singleton identifies both the origin position and the "self" copy, which is uniquely recolored.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="28a6681f" lines="8-23">
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

<reference_puzzle index="8" source="train" puzzle_id="985ae207" lines="82-97">
<description>
Repeat each 3x3 control-center sprite toward the matching solid rectangle until a copy first overlaps that rectangle.

Input structure
- The background is the dominant color.
- Each sprite is a 3x3 foreground object whose eight border pixels have one color and whose center has a different control color.
- For every control color, there is exactly one separate solid rectangular object of that color.

Rule
- Match each sprite to the unique solid rectangle whose color equals the sprite center.
- Copy the whole 3x3 sprite in the cardinal direction of that rectangle, stepping by three cells each time.
- Stop after pasting the first copy whose 3x3 bbox overlaps the rectangle; copied pixels overwrite the rectangle there.

Key insight
The center pixel is not the paint color: it is a destination key, while the entire 3x3 sprite is the motif tiled along the straight path.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="e376de54" lines="41-55">
<description>
Align parallel colored line segments so they all share the same start and end positions as a reference line.

Input structure:
- bg = most common color.
- Several parallel line segments (diagonal, horizontal, or vertical), each a different color, running in the same direction but starting/ending at different positions.

Rule:
- Detect the dominant direction shared by all segments.
- Pick the median segment (by perpendicular offset) as the reference.
- Shift every segment along the shared direction so its start aligns with the reference's start, then trim or extend it to match the reference's length.
- Grid size stays the same; isolated single pixels are untouched.

Key insight: The reference is the median line, not the longest; all segments are snapped to its extent regardless of their original length.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="e3f79277" lines="8-23">
<description>
Replace a corner L with a same-color right-triangle outline in the corresponding corner of an enlarged canvas.

Input structure:
- The input is a square grid with one background color and one foreground color.
- The foreground is a one-pixel-thick L inside a square bbox.
- The L touches exactly one horizontal grid edge and one vertical grid edge, identifying a corner.

Rule:
- Fill a square output canvas of side 3n-2, where n is the input side length, with the background color.
- Double the L leg length to get the triangle side length.
- Draw the two perpendicular triangle legs on the same output corner edges and connect their free endpoints with the diagonal.

Key insight:
- The input L encodes the corner, color, and side length of the triangle outline; the missing side is the diagonal between the L's two free endpoints.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="7b3084d4" lines="253-266">
<description>
Assemble colored rectangular sprites into a single tiled rectangle, anchored by a marker pixel.

Input structure:
- bg = most common color. Several solid-colored rectangular sprites scattered on the grid.
- One sprite has a unique single-pixel marker (different color) at one corner — this is the anchor.

Rule:
- Identify the anchor sprite (contains the marker pixel) and the remaining pieces.
- Rotate the anchor so the marker pixel lands at position (0,0) of the output.
- Pack all remaining pieces (allowing 90° rotations) to tile a complete rectangle with no gaps.

Key insight: The lone marker pixel picks the anchor and fixes the output orientation — it must sit at the top-left corner.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="17b866bd" lines="49-67">
<description>
Move each marker from a lattice corner gap into the adjacent enclosed 12-pixel cavity.

Input structure:
- The grid is a repeated teal lattice around black 12-pixel cavities.
- A marker is a non-black pixel placed at the northwest corner gap of one
  complete 5x5 lattice tile; the marker may itself be teal.

Rule:
- For every marked northwest corner gap, clear that marker to black.
- Fill the 12 black pixels in the tile immediately down-right of the marker
  with the marker color.
- Leave the rest of the lattice unchanged.

Key insight:
- The apparent isolated markers are control pixels for the adjacent cavity;
  teal markers are recognized as extra teal pixels occupying a gap that the
  periodic lattice template says should be black.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="46c35fc7" lines="21-34">
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

<reference_puzzle index="14" source="train" puzzle_id="689c358e" lines="72-88">
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

<reference_puzzle index="15" source="train" puzzle_id="b74ca5d1" lines="157-172">
<description>
Swap each 5x5 sprite with its corner marker color and stamp same-marker composite masks into the matching corner.

Input structure:
- A 30x30 grid has one dominant background color.
- Non-background corner pixels are marker labels.
- Every other foreground color is the body of one 5x5 sprite containing exactly one pixel of a corner marker color.

Rule:
- For each sprite, recolor body pixels to the sprite marker color and recolor the marker pixel back to its body color.
- If a marker lies on the side facing its matching corner and another same-marker sprite has a body pixel at that same local cell, use that visible body color for the marker pixel.
- For each corner marker, overlay the masks of all sprites with that marker color into the corner 5x5 block and paint the visible composite with the marker color.

Key insight:
The corner pixels are labels that group 5x5 sprites by their embedded marker color; each labeled corner shows the grouped sprite silhouette.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="b5ca7ac4" lines="7-21">
<description>
Move bordered 5×5 squares to their designated grid edges by border color.

Input structure:
- bg = most common color.
- Azure (8) and red (2) bordered 5×5 squares with colored interiors, scattered across the grid.

Rule:
- Azure squares → left edge (col 0); red squares → right edge (col W−5).
- Each square keeps its original row.
- Process azure left-to-right, red right-to-left (nearest-to-edge first).
- If a square's rows overlap an already-placed edge square, place it one square-width inward instead.

Key insight: Near-edge-first ordering plus row-overlap determines which square sits at the edge vs. the second column.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="36a08778" lines="8-21">
<description>
Simulate water falling from top-edge sources, pooling on shelves, and cascading off edges.

Input structure:
- bg = most common color; water = rarest non-bg; shelf = other non-bg.
- Sources: water clusters touching top row. Shelves: horizontal segments.

Rule:
- Water falls down from sources, painting bg cells as water.
- On hitting a shelf, it spreads across the shelf span plus one cell past each end.
- From those overflow positions, water resumes falling independently.

Key insight: Each shelf splits a stream into two at its edges, creating a branching waterfall tree.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="2ccd9fef" lines="173-187">
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

<reference_puzzle index="19" source="train" puzzle_id="494ef9d7" lines="16-30">
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

<reference_puzzle index="20" source="train" puzzle_id="a416fc5b" lines="95-106">
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
