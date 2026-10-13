你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="53fb4810" lines="105-119">
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

<reference_puzzle index="2" source="train" puzzle_id="6ffe8f07" lines="49-64">
<description>
Cast yellow horizontal and vertical rays from the teal rectangle until blue blockers stop them.

Input structure:
- A black-background grid containing one solid teal rectangle.
- Blue objects are blockers; other non-background objects may lie in the ray paths.
- Yellow is absent from the input and is the output paint color.

Rule:
- From every row of the teal rectangle, cast rays left and right to the grid edge.
- From every column of the teal rectangle, cast rays up and down to the grid edge.
- A ray stops before a blue pixel; all crossed non-blue, non-teal pixels become yellow.

Key insight:
The teal rectangle defines a full cross-shaped projection, while blue objects are walls and red/background cells are simply paintable space.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="2e65ae53" lines="88-103">
<description>
Fill every lattice slot from the color assigned to that slot position by the other lattices.

Input structure:
- The background is the dominant color.
- The largest foreground color forms several one-pixel-thick rectangular lattice frames.
- The frame lines divide each lattice into the same row-by-column set of rectangular slots; some slots already contain one non-frame color.

Rule:
- Infer a color for each logical slot position from the colored slots across all lattice frames.
- Repaint every slot interior in every lattice with the inferred color for its row/column position.
- Keep the frame lines and all background outside frames unchanged.

Key insight:
- Slot sizes vary by frame, but the complete color pattern is indexed by lattice slot position rather than by absolute pixel size or location.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="f8cc533f" lines="10-25">
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

<reference_puzzle index="5" source="train" puzzle_id="37ce87bb" lines="33-47">
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

<reference_puzzle index="6" source="train" puzzle_id="17829a00" lines="21-36">
<description>
Move each colored object vertically until it touches the matching top or bottom border bar.

Input structure:
- The background is the dominant color.
- The top row is a full-width bar in one foreground color and the bottom row is a full-width bar in another.
- All other foreground pixels are 8-connected monochrome objects whose color matches one of those bars.

Rule:
- Top-bar-colored objects move upward so their bbox top is row 1.
- Bottom-bar-colored objects move downward so their bbox bottom is the row above the bottom bar.
- A downward-moving one-column segment whose landing overlaps its source becomes the continuous vertical segment swept between them.

Key insight:
The horizontal bars are destination anchors: color determines which border an object slides toward, while columns and object shape are otherwise preserved.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="3ad05f52" lines="128-142">
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

<reference_puzzle index="8" source="eval" puzzle_id="97d7923e" lines="81-94">
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

<reference_puzzle index="9" source="train" puzzle_id="230f2e48" lines="19-38">
<description>
Fold each straight red rod at its black marker, leaving the gray-anchored side fixed.

Input structure:
- The background is the dominant color.
- Each foreground object is a straight horizontal or vertical 4-connected rod.
- Each rod contains many pixels of one line color, one endpoint marker, and one
  interior control marker.

Rule:
- For every rod, keep the segment from the endpoint marker through the control
  marker unchanged.
- Move the line-color tail on the far side of the control marker to a
  perpendicular ray starting next to that marker.
- Choose the perpendicular direction with more space to the grid edge.

Key insight:
- The endpoint marker identifies the fixed side of the rod, while the control
  marker is the hinge; the tail folds into the larger open half-plane.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="2d0172a1" lines="209-224">
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

<reference_puzzle index="11" source="train" puzzle_id="afe3afe9" lines="106-119">
<description>
Convert each hollow 3x3 symbol to one pixel and slide the pixels clockwise from the marked edge.

Input structure
- The grid has a background, one full-edge marker line, and monochrome hollow 3x3 foreground symbols.
- The symbols lie on a coarse row/column lattice; fully empty lattice rows or columns are absent from the output.

Rule
- Replace each hollow 3x3 symbol with one pixel of its color in the corresponding coarse-grid cell.
- The marker edge gives the compaction direction by clockwise border travel: top -> right, right -> down, bottom -> left, left -> up.
- Preserve the order of non-background pixels while sliding them as far as possible in that direction.

Key insight
- The large picture is a scaled symbolic grid, and the blue edge marker acts as a gravity direction rather than as output content.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="981add89" lines="10-29">
<description>
Extend each top-row control pixel downward as a vertical toggle ray in its own color.

Input structure:
- The background is the dominant color.
- Non-background pixels on the top row are control pixels, each identifying
  one column and one ray color.
- Foreground regions below may intersect these control columns.

Rule:
- For every top-row control pixel, process the cells below it in the same
  column down to the bottom edge.
- A cell that already has the control color is erased to the background.
- Every other cell on that downward ray is painted with the control color.
- The top-row control pixels themselves remain unchanged.

Key insight:
The top row is a set of vertical ray launchers; each ray toggles only cells
matching its own color and overwrites all other colors along that column.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="8f3a5a89" lines="22-36">
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

<reference_puzzle index="14" source="train" puzzle_id="fc10701f" lines="20-35">
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

<reference_puzzle index="15" source="train" puzzle_id="dc46ea44" lines="11-25">
<description>
Move the lower sprite and its pink guide into the upper panel, aligning the sprite to the guide's lower-left branch.

Input structure
- The grid has one dominant background color and one full-width separator row.
- Below the separator are exactly two foreground roles: a five-pixel guide marker and one monochrome sprite.
- The guide marker has normalized shape {(0,2), (1,2), (2,0), (2,2), (3,1)} and starts immediately below the separator.

Rule
- Clear the canvas to the background, preserving the separator row.
- Translate the guide marker upward so its top row becomes row 0.
- Move the sprite, preserving its color and shape, so the sprite bbox bottom-right aligns to the translated guide's lower-left branch.

Key insight
- The marker supplies the destination anchor: its lower-left branch cell is where the moved sprite's bbox bottom-right must land.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="22208ba4" lines="12-27">
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

<reference_puzzle index="17" source="train" puzzle_id="bd283c4a" lines="7-23">
<description>
Pack all pixels into color-frequency runs along bottom-up columns.

Input structure
The input is a rectangular grid whose colors are all foreground material;
only the number of pixels of each color matters. In the observed task
instances, every present color has a distinct pixel count.

Rule
Count the pixels of each color and order the colors from most frequent to
least frequent. Rewrite a same-size grid by placing each color's full count
as one contiguous run while scanning columns from left to right and, within
each column, from bottom to top.

Key insight
The output preserves the multiset of colors but discards all original
positions, turning the grid into a frequency-sorted column packing.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="963c33f8" lines="86-108">
<description>
Drop the columns of the top marker downward, stopping blue-ended columns on grey obstacles and letting maroon-ended columns pass to the bottom.

Input structure
- The background is the dominant color.
- The most numerous non-background color forms stationary grey obstacle pixels.
- The other two colors form a single 3x3 top-edge marker whose top two rows
  are maroon and whose bottom row contains maroon or blue control pixels.

Rule
- Erase the top marker.
- Treat each marker column as a vertical drop retaining its three colors.
- A column whose bottom control pixel is blue stops immediately above the
  first grey pixel below that column, or bottom-aligns if none exists.
- Contiguous maroon-ended columns pass through obstacles and bottom-align;
  if such a run has a clear down-right slide from a grey ledge, shift the
  run one column right before bottom-aligning.
- When a maroon-ended drop lands through a grey bar on the bottom edge, clear
  the rest of the marker-width gap in that bottom bar.

Key insight
- The bottom row of the top marker is a control strip: blue cells mean
  "stick to the first obstacle," while maroon cells mean "fall through."
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="16de56c4" lines="9-34">
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

<reference_puzzle index="20" source="train" puzzle_id="30f42897" lines="51-66">
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

