# Issue tracker: Hybrid (local markdown drafts → GitHub Issues)

Work for this repo moves through two stages:

1. **Draft / research stage — local markdown.** Exploratory efforts, research
   questions, and unfinished specs live as files under `.scratch/<feature-slug>/`
   at the *workspace root* — the directory containing this repo checkout
   (e.g. `C:\cursor\DMDH\.scratch\`). If no workspace root exists (standalone
   clone), use `.scratch/` at the repo root.
2. **Published stage — GitHub Issues.** When a spec is actionable, publish it as
   a GitHub issue on `KasperHonore/Foam-Agent`. Triage, labels, and state all
   happen on GitHub. Record the published issue number (`Published: #NN`) near
   the top of the local file so it isn't re-published.

## Local conventions (draft stage)

- One effort per directory: `.scratch/<feature-slug>/`
- The PRD is `.scratch/<feature-slug>/PRD.md`
- Draft issues are `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Supporting material goes in `.scratch/<feature-slug>/assets/`
- Triage state is a `Status:` line near the top of each draft (see `triage-labels.md`
  for the role strings)
- Comments and conversation history append under a `## Comments` heading

## GitHub conventions (published stage)

Use the `gh` CLI. Inside the clone, `gh` infers the repo from `git remote -v`;
from the workspace root, pass `-R KasperHonore/Foam-Agent`.

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc
  for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by
  `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`
  with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

## Pull requests as a triage surface

**PRs as a request surface: no.** `/triage` reads only issues.

## When a skill says "publish to the issue tracker"

Create a GitHub issue on `KasperHonore/Foam-Agent`, then write `Published: #<n>`
into the local draft if one exists.

## When a skill says "fetch the relevant ticket"

- Given a `#number` (or bare number): `gh issue view <number> --comments`.
- Given a file path: read the file. If it has a `Published: #<n>` line, also
  fetch the GitHub issue for the latest state.

## Wayfinding operations

Used by `/wayfinder`. Wayfinding happens in the **local draft stage** (this is
the existing `.scratch/foam-agent-next/` convention):

- **Map**: `.scratch/<effort>/map.md` — the Notes / Decisions-so-far / Fog body.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`,
  with the question in the body. A `Type:` line records the ticket type
  (`research`/`prototype`/`grilling`/`task`); a `Status:` line records
  `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked
  when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open,
  unblocked, and unclaimed; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set
  `Status: resolved`, then append a context pointer to the map's
  Decisions-so-far in `map.md`. If the resolution produced an actionable spec,
  publish it to GitHub and note `Published: #<n>`.
