# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

Edit the right-hand column to match whatever vocabulary you actually use.

## Repo-specific notes

- `ready-for-agent` and `wontfix` already exist as GitHub labels on
  `KasperHonore/Foam-Agent`. `needs-triage`, `needs-info`, and
  `ready-for-human` do not yet — create each on first use with
  `gh label create <name> --description "<meaning from the table>"`.
- The existing `needs-repro` label ("Can't act until reproduced") is **not**
  the `needs-info` role. It is a narrower, bug-specific tag and can coexist
  with (or accompany) `needs-info`.
- In the local draft stage (`.scratch/`), record the role string on the
  `Status:` line of the draft file instead of a GitHub label.
