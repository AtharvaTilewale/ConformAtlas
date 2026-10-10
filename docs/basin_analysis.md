# Basin Analysis & Representative Structure Extraction

ConformAtlas automatically detects energy basins and extracts verified full-atom structures corresponding to each conformational state.

---

## 1. Topological Watershed Segmentation

Rather than simply finding local minimum pixels, ConformAtlas uses **topological watershed segmentation**:
1. Candidate minimum energy seeds are identified as local peaks in probability density (peaks on the grid border count; every connected region of sampled space receives at least one seed).
2. The watershed algorithm segments the continuous energy landscape into catchment basins bounded by energy ridges.
3. Basins separated by a barrier lower than `--min-barrier` (default 1 kT = RT at the analysis temperature, 2.49 kJ/mol at 300 K) are merged, shallowest separation first. This removes the sub-basins that watershed produces from counting noise inside one broad well. Lower the threshold to resolve finer substates, raise it to be more conservative.
4. Every occupied grid cell is assigned to its catchment basin; isolated occupied cells inherit the nearest basin.
5. Frames are assigned to the basin corresponding to their $(PC_1, PC_2)$ coordinates.

Grid resolution matters: with fewer than about two frames per occupied bin the landscape is dominated by counting noise and ConformAtlas prints a warning. Reduce `--bins`, enable `--smooth-sigma` (1.0 is a reasonable start), or sample longer.

---

## 2. Conformational States

* **State Labeling**: Basins are sorted by population descending and labeled `State A` (dominant state), `State B` (second dominant), etc.
* **Population Filtering**: Basins containing $\ge 5\%$ of frames (`--min-state-population 0.05`) are classified as major states. Low-occupancy basins are collected into the unassigned/minor fraction.
* **Population Sum**: $\sum \text{Major States} + \text{Unassigned} = 100\%$.

---

## 3. Representative Structure Extraction

For each state, ConformAtlas extracts a verified real frame from the original full trajectory:
1. Identifies the minimum free energy coordinates $(PC_{1, \min}, PC_{2, \min})$ of the basin.
2. Identifies all frames assigned to this state.
3. Computes the Euclidean distance $d = \sqrt{(PC_1 - PC_{1, \min})^2 + (PC_2 - PC_{2, \min})^2}$.
4. Selects the frame with minimal distance and writes `states/State_X/representative.pdb`.
5. Writes `states/State_X/metadata.json` recording the exact replicate, frame index, time (ps/ns), and coordinates.
