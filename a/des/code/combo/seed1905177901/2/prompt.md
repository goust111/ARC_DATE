你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="412b6263" lines="39-54">
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

<reference_puzzle index="2" source="train" puzzle_id="f18ec8cc" lines="71-86">
<description>
Reorder the vertical color-major blocks while preserving every block's internal pixels.

Input structure
The grid is divided into vertical blocks, each with a unique dominant color.
Small marker pixels from other block colors may appear inside the blocks.

Rule
If adjacent shared-color marker rows read exactly as the current block order
from right to left, reverse the blocks. Otherwise move the leftmost block to
the far right and shift the remaining blocks left.

Key insight
The transformation acts on whole vertical blocks, so marker pixels move with
their source block instead of being recolored or independently rearranged.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="edcc2ff0" lines="47-63">
<description>
Keep only the lower-panel colors named by the top markers and expand each marker into an object-count bar.

Input structure
A black instruction strip sits above a non-black rectangular scene panel.  The strip contains one-cell markers
in the first column, and each marker color is a candidate object color in the lower panel.  The lower panel has
a unique dominant background color and scattered 4-connected foreground objects.

Rule
In the lower panel, preserve objects whose colors appear as top markers and recolor every other foreground
object to the panel background.  In the instruction strip, replace each marker by a horizontal run from column 0
whose length is the number of 4-connected lower-panel objects of that marker color; erase markers with count 0.

Key insight
The top strip is a color filter plus a compact count summary: marker colors select which lower-panel objects
survive, and their bars report the number of surviving objects by color.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="46c35fc7" lines="21-34">
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

<reference_puzzle index="7" source="train" puzzle_id="30f42897" lines="51-66">
<description>
Repeat the single colored border segment around the rectangular border, leaving one segment-length gap between copies.

Input structure:
- The background is the dominant color.
- There is exactly one foreground color.
- All foreground pixels form one contiguous segment on the rectangular border, possibly crossing the cycle start.

Rule:
- Unwrap the grid border clockwise into a cyclic path.
- Let n be the length of the foreground segment.
- Copy the segment with the same color every 2*n border positions around the cycle.

Key insight:
- The foreground segment is not extended by rows or columns directly; it is a motif on the perimeter cycle, repeated with an equal-sized gap.
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

<reference_puzzle index="9" source="eval" puzzle_id="53fb4810" lines="105-119">
<description>
Extend marker tiles outward from '+' shapes by repeating them to the grid edge.

Input structure:
- bg = most common color.
- A dominant non-bg color forms several '+'-shaped objects.
- Small colored patches (1–few cells) sit adjacent to each arm of a '+'.

Rule:
- For each marker tile touching a '+', determine which side of the '+' it sits on.
- Project the tile away from the '+' in that direction, repeating the tile pattern periodically until the grid boundary.
- '+' pixels are never overwritten.

Key insight: Each marker's attachment side determines its extension direction; the pattern repeats periodically, not as a single copy.
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

<reference_puzzle index="11" source="train" puzzle_id="6165ea8f" lines="74-90">
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

<reference_puzzle index="12" source="train" puzzle_id="9caf5b84" lines="28-45">
<description>
Recolor every pixel whose color is not one of the two most frequent input colors to orange.

Input structure:
- The grid contains at least three colors.
- Orange is absent from the input and is used only as the output replacement color.
- Exactly two colors occupy the top two frequency ranks; they may tie with each
  other, but no third color ties them at the cutoff.

Rule:
- Preserve every pixel whose color is one of the two most frequent input colors.
- Replace every other pixel with orange.

Key insight:
- The visible distinction is not the numeric identity of the colors but their
  population: the two majority colors remain, while all minority colors merge
  into the new orange color.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="2b83f449" lines="10-22">
<description>
Simulate tokens falling through a pipe network after gates rotate open.

Input structure:
- Wall (0), bg (pipe fill), gate (horizontal 1×3 bars), token (small edge pixels).

Rule:
- Rotate each gate into a vertical blocker; gate cells become bg.
- Tokens drop downward through bg, choosing the extreme column for their side (left→min, right→max).
- Tokens settle at true pipe ends (adjacent to wall/boundary), packing inward.

Key insight: Tokens select drop columns by entry side and land at wall-bounded endpoints, ignoring blocker-bounded ones.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="dd6b8c4b" lines="9-22">
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

<reference_puzzle index="15" source="eval" puzzle_id="8f3a5a89" lines="22-36">
<description>
Paint an inner border around the region reachable from a corner marker pixel.

Input structure:
- bg = most common color; wall color (blue) forms barriers dividing the grid.
- A unique single pixel (magenta) sits on one corner — the start marker.
- Some wall-colored pixels are structural (border-touching, enclosing regions); others are decorations.

Rule:
- Flood-fill from the marker corner to find all reachable bg cells, blocked by structural walls.
- Erase wall-colored decorations hidden behind structural walls (not reachable from marker side).
- Paint orange (color 7) on every reachable bg cell adjacent to a wall or grid edge — forming a 1-cell inner border.

Key insight: Only border-connected wall components act as real barriers; interior wall fragments are decorations to be removed.
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

<reference_puzzle index="17" source="eval" puzzle_id="898e7135" lines="7-20">
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

<reference_puzzle index="18" source="eval" puzzle_id="4c7dc4dd" lines="7-21">
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

<reference_puzzle index="19" source="eval" puzzle_id="cb2d8a2c" lines="13-26">
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

<reference_puzzle index="20" source="eval" puzzle_id="3dc255db" lines="37-53">
<description>
Eject content pixels from cup-shaped frames outward along each cup's tail direction.

Input structure:
- bg = background. Hollow cup/Y/funnel frames, each a unique border color.
- Each cup has a convergent tail spike on one side and an open mouth on the opposite side.
- Content pixels (a single non-border color per cup) sit inside the cup's mouth interior.

Rule:
- For each cup, identify its tail spike and tail direction.
- Erase all content pixels from the mouth interior.
- Draw a straight line of content-colored pixels from the spike outward in the tail direction,
  one pixel per erased content pixel; cells beyond the grid edge are dropped.

Key insight: The tail spike is the unique convergent point whose outward direction is opposite
the mouth opening where the content lives.
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
