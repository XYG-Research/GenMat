# Surface Screen Module

`genmat surface-screen` scans CIF files from a directory, enumerates low-complexity slabs, records raw surface descriptors for every inequivalent termination, optionally searches compensated slabs for asymmetric/polar cuts, and can run a fast MLIP-based sanity screen before passing slabs to adsorption workflows.

## Entry point

```powershell
genmat surface-screen --input-dir . --out-dir output/surface_screen --recursive
```

Default behavior:

- scans the current directory for `*.cif`
- enumerates symmetry-distinct low-index facets with `max(|h|,|k|,|l|) <= 2`
- filters by `d_hkl`, stoichiometric-repeat slab thickness, vacuum thickness, and surface cell area
- enumerates inequivalent terminations along the slab normal
- assigns each termination both a `raw_class` and a `final_class`
- treats raw polarity/asymmetry as a descriptor instead of a hard rejection
- can automatically search compensated slabs with thickness capped at `15 A` by default
- exports an additional modeling-friendly slab representation with the slab centered along `z`
- can optionally run MLIP pre-relaxation with `eSEN-30M-MPtrj`

## Layered pipeline

### Step 1. Low-index facet enumeration

This stage is intentionally lattice-agnostic. It does not hard-code `(100)/(110)/(111)` by crystal system.

For each CIF:

1. build symmetry-distinct Miller indices using `pymatgen.get_symmetrically_distinct_miller_indices`
2. keep only facets satisfying:
   - `max(|h|,|k|,|l|) <= max_index`
   - `d_hkl >= min_d_hkl`
   - `min_surface_area <= slab area <= max_surface_area`
3. choose slab thickness by stoichiometric repeats:
   - prefer `3` stoichiometric repeats
   - if that is too thick, use `2` repeats
   - never go below `2` repeats
   - if the actual minimum thickness among the facet's `2`-repeat terminations exceeds `max_slab_thickness_a`, keep `2` repeats and mark thickness override
4. generate slabs with the chosen stoichiometric-repeat thickness and the requested vacuum size

If a preferred `3`-repeat termination is too thick but the same facet still has valid `2`-repeat terminations below the thickness limit, the workflow automatically falls back to the closest `2`-repeat termination instead of discarding the surface outright. If the retained `2`-repeat termination still exceeds the thickness limit, the workflow can make a final fallback to a matching `1`-repeat termination from the same facet. The fallback match is selected by comparing cleavage shift, top/bottom layer formulas, top-two-layer formulas, asymmetry, and surface atomic density.

This yields a structure-adaptive low-complexity facet set instead of a fixed facet list.

### Step 2. Inequivalent termination enumeration and raw descriptors

For each retained facet:

1. enumerate all cutting shifts along the facet normal
2. generate one slab per shift
3. merge duplicate terminations by a structure signature
4. record termination descriptors:
   - top-layer formula
   - bottom-layer formula
   - top-two-layer formula
   - bottom-two-layer formula
   - symmetric slab flag
   - dipole-risk proxy
   - top/bottom composition asymmetry
   - surface roughness
   - top/bottom surface atom density
   - mean coordination loss and severe-loss fraction

The raw termination labels are:

- `raw_good`
- `raw_polar_or_asymmetric`
- `raw_broken_skeleton`
- `raw_broken_skeleton_and_polar`
- `raw_high_roughness`

Important policy change:

- `raw_polar_or_asymmetric` is **not** a hard rejection
- raw polarity/asymmetry is treated as a descriptor for multicomponent surfaces
- clear skeleton breaking and severe geometric damage still remain rejection signals later

### Step 2b. Automatic compensated slab search

For raw polar/asymmetric terminations, the module can automatically search compensated alternatives.

Current search route:

- `Tasker2` correction

Constraints:

- only search when the raw slab is flagged as polar/asymmetric
- only keep compensated candidates with slab thickness `<= max_compensated_slab_thickness_a`
- default thickness cap is `15 A`
- the workflow does **not** reduce `min_slab_size` to force compensation

The best compensated candidate is selected by a score based on asymmetry, polarity proxy, coordination loss, severe undercoordination, and roughness.

### Step 3. Final classification and optional MLIP screen

Enable with:

```powershell
genmat surface-screen --input-dir . --out-dir output/surface_screen --run-mlip
```

The module uses the existing GenMat MLIP helpers and can resolve the alias:

- `eSEN-30M-MPtrj`

If the checkpoint is not provided manually, the code can use the local ADSPP checkpoint naming via the existing MLIP alias mapping.

Current behavior:

- freeze the middle slab layers
- relax only the outer `N` layers per side
- compute:
  - maximum atomic displacement
  - surface-layer RMSD
  - interlayer spacing change
  - cross-layer reconstruction flag

Before MLIP, the module assigns one `final_class`:

- `direct_surface_candidate`
- `polar_surface_candidate`
- `compensated_surface_candidate`
- `compensated_polar_surface_candidate`
- `rejected_broken_skeleton`
- `rejected_high_roughness`
- `rejected_too_thick`

If MLIP is enabled, any pre-accepted slab can still become:

- `rejected_after_mlip`

## Recommended settings for complex intermetallics

For high-entropy or quinary intermetallics:

```powershell
genmat surface-screen `
  --input-dir . `
  --out-dir output/surface_screen `
  --recursive `
  --max-index 2 `
  --min-d-hkl 1.2 `
  --min-slab-size 12 `
  --min-vacuum-size 15 `
  --max-surface-area 160
```

If adsorption enumeration is still too large, tighten:

- `--min-d-hkl 1.5`
- `--max-surface-area 120`
- `--max-terminations-per-facet 6`

## Output files

The command writes:

- `facet_screening.csv`
  - one row per candidate facet
  - includes Miller index, `d_hkl`, slab area, and whether the facet passed Step 1
- `termination_screening.csv`
  - one row per inequivalent termination from the preferred stoichiometric-repeat slab set
  - includes raw descriptors, fallback layer-source/layer-target fields, compensation search results, final descriptors, `final_class`, MLIP screen fields, canonical family mapping fields, and final keep flag
- `structure_summary.csv`
  - one row per input CIF
  - counts retained facets, raw terminations, and canonical termination families
- `failures.csv`
  - input files that could not be processed
- `surface_screen_summary.json`
  - global counts and the exact config used
- `accepted_slabs/`
  - retained slab CIFs after screening
  - only canonical representative slabs are exported here
- `modeling_slabs/`
  - rotated/exported slab CIFs with the slab centered along `z`
  - attempts an integer in-plane supercell search to make the modeling cell orthogonal or near-orthogonal
  - only canonical representative slabs are exported here

## Practical interpretation

- Use `facet_screening.csv` to understand how many low-complexity facets each bulk structure contributes.
- Use `termination_screening.csv` as the main handoff table for adsorption workflows.
- Use the canonical family fields in `termination_screening.csv` to distinguish original termination enumeration from unique adsorption-ready representative slabs.
- Treat `direct_surface_candidate` as the highest-priority surface set.
- Treat `compensated_surface_candidate` as high-value surfaces recovered from asymmetric raw cuts.
- Treat `polar_surface_candidate` and `compensated_polar_surface_candidate` as exploratory but still physically relevant multicomponent surfaces.
- Treat `rejected_broken_skeleton` as slabs that are mathematically cleavable but not good adsorption targets.
- Treat `rejected_too_thick` as slabs that violate the global thickness cap **unless** the run explicitly allowed the stoichiometric-thickness override.
- Use `canonical_is_representative = True` when you want one unique slab per termination family; use all mapped rows when you want to preserve the provenance of every original cleavage choice.

## Notes

- The current dipole criterion is a chemistry/symmetry proxy, not a full electrostatic dipole from charge-decorated slabs.
- The Windows build avoids `pymatgen.get_slabs()` and enumerates termination shifts directly, because the upstream helper path is unstable in this environment.
- The module is designed to be conservative for complex multicomponent slabs; simple systems such as B2 AlNi already produce retained direct or compensated candidates in smoke tests.
