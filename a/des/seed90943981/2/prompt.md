你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="88bcf3b4" lines="229-241">
<description>
Swing an irregular line segment around a nearby obstacle sprite, preserving its pixel count.

Input structure:
- bg, bar color (straight lines perpendicular to the grid edge, one end on the border), segment color (irregular stroke attached to bar's inner endpoint), obstacle color (sprite near the bar terminal).

Rule:
- Each bar is perpendicular to the edge it touches; its inner (terminal) endpoint has an attached irregular segment.
- The segment detaches and re-routes: it travels from the terminal toward the nearest obstacle, hugs the obstacle's boundary, then exits at a 45° diagonal.
- Total painted pixels equal the original segment length.

Key insight: The segment behaves like a rope swung around the obstacle — attach at the nearest point, trace the skin, exit diagonally, conserving length.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="e376de54" lines="41-55">
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

<reference_puzzle index="3" source="eval" puzzle_id="247ef758" lines="35-46">
<description>
Move colored sprites from the left strip into a framed area at marker-defined targets.

Input structure:
- bg background; rectangular frame on right with colored markers on its edges; sprites in a narrow left strip separated by a constant-color divider column.

Rule:
- For each color, target rows/cols are where both opposite frame edges share that color. Targets are the Cartesian product.
- Erase each sprite from the left and stamp it centered at every target. Higher sprites cover lower ones.

Key insight: One sprite replicates to a grid of row × column marker intersections.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="53fb4810" lines="105-119">
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

<reference_puzzle index="5" source="eval" puzzle_id="2c181942" lines="7-21">
<description>
Attach monochrome sprites to the matching sides of a central hollow square.

Input structure:
- bg = background. A hollow NxN square (hub) with bg corners/interior and four differently-colored edges (top, bottom, left, right).
- 2-4 monochrome sprites scattered on the grid, each matching one hub edge color.

Rule:
- For each hub edge color, find the matching sprite outside the hub.
- Rotate the sprite (0/90/180/270°) so its contact edge has exactly N-2 contiguous cells aligned with the hub side, and the rotated sprite is mirror-symmetric along the contact-edge axis (UD for left/right, LR for top/bottom).
- Translate it to adjoin that hub side. 
- Clear everything else to bg; output is hub + attached sprites only.

Key insight: Each edge color uniquely pairs a sprite to a side; symmetry along the contact-edge axis disambiguates the correct rotation.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="6e453dd6" lines="39-52">
<description>
Slide black shapes rightward to a grey separator, then shoot red rays from enclosed holes.

Input structure:
- A vertical grey (5) separator column divides the grid into left and right halves.
- Black shapes sit on the left side against a background color (bg).

Rule:
- Translate each black connected component rightward until its right edge is flush with the separator.
- After sliding, find enclosed bg regions (holes) within the black shapes that touch the column just left of the separator.
- For each row spanned by such a hole, emit a red ray across the entire right half of the grid.

Key insight: Only holes adjacent to the separator trigger rays — the sliding creates enclosures that act as "nozzles" projecting rightward.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="80a900e0" lines="69-82">
<description>
Extend diagonal rays outward from outer-border segments adjacent to diamond shapes on a checkerboard.

Input structure:
- Checkerboard bg (two most common colors).
- Hollow diamond outlines (8-pixel) in a non-bg color (inner border).
- Short diagonal line segments in another non-bg color adjacent to each diamond edge (outer border).

Rule:
- For each outer-border segment, determine the outward direction (perpendicular to the segment, away from the diamond).
- From each endpoint of the segment, cast a diagonal ray outward, painting over bg pixels with the segment's color until hitting a non-bg pixel or the grid edge.

Key insight: Rays emit only from the two endpoints of each outer-border segment, not every pixel, and travel diagonally away from the diamond.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="b99e7126" lines="34-48">
<description>
Scale a minority cell's inner pattern up to the cell-grid level, completing a fractal overlay.

Input structure:
- Grid divided into NxN cells by separator lines of a single color.
- Most cells share an identical interior pattern (majority); a few have a distinct pattern (minority).

Rule:
- Extract the inner pixel pattern from the minority cell.
- Treat each pixel as a cell-sized stamp: active pixels → minority pattern, inactive → majority pattern.
- Position this scaled-up pattern on the cell grid so existing minority cells align with active pixels.
- Fill in all remaining active positions with the minority pattern.

Key insight: The minority cell's interior is a miniature blueprint of where minority cells should appear at the grid scale — fractal self-similarity.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="89565ca0" lines="120-136">
<description>
Convert each colored partitioned frame into a horizontal bar whose length is its number of enclosed regions.

Input structure:
- Black background with several axis-aligned colored line drawings.
- The line drawings are damaged by scattered pixels of one non-background noise color and by overlaps with other drawings.
- Each real drawing is a rectangular partitioned frame enclosing one or more background regions.

Rule:
- For each non-background color, reconstruct its horizontal and vertical wall segments from dense row/column evidence.
- Count the 4-connected background regions fully enclosed by those reconstructed walls.
- The color with no enclosed region is the noise/pad color.
- Sort drawing colors by enclosed-region count; equal counts are ordered by which reconstructed frame covers the other.

Key insight:
- The scattered color is not an object; it only pads the histogram, while each true object's room count is stable after rectilinear wall reconstruction.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="13e47133" lines="56-69">
<description>
Fill partitioned regions with concentric colored bands derived from marker pixels.

Input structure:
- Grid partitioned into regions by wall-colored lines.
- Each region has a dominant base color and a few isolated marker pixels of other colors at varying depths from the region boundary.

Rule:
- For each region, read marker colors ordered by their Chebyshev distance from the boundary to build a color palette.
- If no marker sits on the boundary itself, the base color leads the palette.
- Paint every cell with palette[distance mod palette_length], producing repeating concentric rectangular frames that radiate inward.

Key insight: Each marker's depth from the boundary defines its position in the cyclic palette; the pattern then tiles inward automatically.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="a25697e4" lines="9-23">
<description>
Dock movable tools into border-anchored frames by filling cavities.

Input structure:
- bg = most common color.
- Border-touching components cluster into anchor frames with interior bg cavities.
- Non-border 2-color components are movable tools.

Rule:
- Match each tool to a frame whose cavity size equals one of its color-region counts.
- Rotate/flip the tool so one color fills the cavity exactly; the other stays outside.
- If the docking orientation uses a reflection, swap the tool's two colors.

Key insight: Cavity size determines tool–frame pairing; reflection triggers color swap.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="7666fa5d" lines="12-26">
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

<reference_puzzle index="13" source="eval" puzzle_id="16de56c4" lines="9-34">
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

<reference_puzzle index="14" source="eval" puzzle_id="8b7bacbf" lines="42-56">
<description>
Fill enclosed regions along a line seeded by unique border pixels.

Input structure:
- bg = most common color.
- Colored lines/drawings on the grid.
- Unique single-pixel colors on the grid border mark seed endpoints of each line.

Rule:
- Each border seed pixel connects (8-connected) to a line drawing.
- Find closed bg cavities whose boundary touches that line.
- Flood-fill each such cavity with the seed pixel's color.

Key insight: Only cavities reachable through the seed's own line get filled; the seed's unique color acts as a per-line fill label.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="e87109e9" lines="68-83">
<description>
Emit four teal beams from a square, turning at colored obstacles per a legend.

Input structure:
- Top band: grey (5) legend with colored vertical bars in cells; bar position (left or right) encodes turn direction.
- Below: main grid with bg, a solid teal (8) square, and colored obstacle blocks.

Rule:
- Remove the legend from the output.
- From the teal square, shoot a beam in each cardinal direction (as a sliding copy of the square).
- Each beam paints teal over bg cells as it moves.
- On hitting an obstacle of color C, consult the legend: bar on left → turn counterclockwise, bar on right → turn clockwise. Continue in the new direction.
- Beams stop at grid edges.

Key insight: The legend's bar position (left vs right within its cell) is the encoding for turn direction at each obstacle color.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="271d71e2" lines="9-22">
<description>
Move bordered containers toward same-colored destination markers, charging gray cells to orange.

Input structure:
- bg = 6 (magenta). Black-bordered (0) rectangles with gray (5) / orange (7) interiors.
- A head sprite (9) sits adjacent to one container edge; a matching destination sprite lies along the movement axis.

Rule:
- Each container translates toward its destination along the head-to-dest axis.
- Steps = min(head-to-dest distance, number of gray interior cells).
- Gray cells become orange from the head-facing side inward, one per step.

Key insight: Gray cell count caps movement — a container stops short if it runs out of gray cells to charge.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="b10624e5" lines="24-39">
<description>
Replicate a motif across quadrants with scaling and reflection keyed to red anchor blocks.

Input structure:
- Grid divided into four quadrants by a full-row and full-column divider of a single non-bg color.
- Each quadrant contains a red (2) square anchor block of varying size.
- The top-left quadrant also has a colored motif attached to its red anchor.

Rule:
- Use the top-left quadrant's motif as the source pattern.
- For each other quadrant, scale the motif by the ratio of that quadrant's red block size to the source red block size.
- Reflect horizontally for right-side quadrants, vertically for bottom quadrants (both for bottom-right).
- Align the scaled/reflected motif so its red region overlaps the target anchor, then stamp non-bg pixels.

Key insight: Red block size encodes the scale factor; quadrant position encodes the reflection axes.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="5961cc34" lines="8-21">
<description>
Build a red ray network by activating blue blobs via directional green ports.

Input structure:
- bg = most common color. Large blue blobs with adjacent green pixels marking port directions.
- Start marker: yellow pixel atop a short red segment at grid bottom.

Rule:
- Extend a ray upward from start; on hitting a blue blob, recolor it red.
- Activated blobs emit thick rays through green ports until boundary or another blob.
- Cascade until no new blobs are reached.

Key insight: Green pixels encode ray direction and width; the network activates as a chain reaction from one start point.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="2b83f449" lines="10-22">
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

<reference_puzzle index="20" source="eval" puzzle_id="8f215267" lines="7-21">
<description>
Count same-color patterns per frame and represent the tally as dots inside.

Input structure:
- bg = most common color.
- Hollow rectangular frames in various colors.
- Small monochrome pattern sprites scattered outside frames.

Rule:
- Remove all small patterns from the grid.
- For each frame of color C, count how many patterns share color C.
- Draw that many dots along the frame's center row, spaced by 2, starting from the right.

Key insight: Frames act as tally displays — scattered same-color patterns are "absorbed" and their count is shown as interior dots.
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
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
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

