# AskingMe Agent Skills

AskingMe Agent loads enterprise policy skills from `ASKINGME_SKILLS_DIR` at startup and injects matching rules into the target agent's system prompt.

## Current Domains

```text
skills/general_policy/SKILL.md  # General policy Q&A, version clarification, and routing
skills/expense_policy/SKILL.md  # Expenses, travel, procurement, and approval boundaries
skills/hr_policy/SKILL.md       # Leave, attendance, benefits, onboarding, and offboarding
skills/access_policy/SKILL.md   # Accounts, roles, and data-access policies
```

Front matter fields:

- `name`: skill name
- `description`: responsibility and scope
- `keywords`: comma-separated request-matching terms
- `agents`: one or more of `general`, `expense`, `hr`, and `access`
- `priority`: matching priority when several skills apply

## Authoring Principles

- Every policy conclusion must trace to knowledge-base content or an explicit skill rule.
- Keep each skill focused on one policy domain.
- State scope, effective version, required materials, approvers, and exception handling.
- Never include real employee data, credentials, keys, or confidential company information.
