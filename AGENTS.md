# AGENTS.md — Kideta Workflow

This file enforces a skill-driven execution model powered by the `skill` tool.

## Core Rules

- If a task matches a skill, you **must** invoke it using the `skill` tool
- Skills are located in `skills/<skill-name>/SKILL.md` and `.claude/skills/<skill-name>/SKILL.md`
- Never implement directly if a skill applies
- Always follow the skill instructions exactly

## Intent → Skill Mapping

| Intent | Skill(s) |
|--------|----------|
| Feature / new functionality | `spec-driven-development` → `incremental-implementation` + `test-driven-development` |
| Planning / breakdown | `planning-and-task-breakdown` |
| Bug / failure / unexpected behavior | `debugging-and-error-recovery` |
| Code review | `code-review-and-quality` |
| Refactoring / simplification | `code-simplification` |
| API or interface design | `api-and-interface-design` |
| Performance optimization | `performance-optimization` |
| Security | `security-and-hardening` |
| Documentation | `documentation-and-adrs` |
| Shipping / release | `shipping-and-launch` |

## Lifecycle Mapping (Implicit Commands)

| Phase | Skill |
|-------|-------|
| DEFINE | `spec-driven-development` |
| PLAN | `planning-and-task-breakdown` |
| BUILD | `incremental-implementation` + `test-driven-development` |
| VERIFY | `debugging-and-error-recovery` |
| REVIEW | `code-review-and-quality` |
| SHIP | `shipping-and-launch` |

## Execution Model

For every request:
1. Determine if any skill applies (even 1% chance)
2. Invoke the skill using the `skill` tool
3. Follow the skill workflow strictly
4. Only proceed to implementation after required steps are complete

## Anti-Rationalization

Incorrect thoughts you **must ignore**:
- "This is too small for a skill"
- "I can just quickly implement this"
- "I'll gather context first"
- "This doesn't need formal process"

Correct: always check for and use skills first.
