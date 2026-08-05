# Repository guidance

## Project

Triangles explores algorithms that approximate target RGB images with an ordered, fixed-size collection of translucent triangles. The benchmark and scoring contract are documented in `README.md`. Always read this file as context.

## Development Principles

Avoid complexity. Strive for simple, minimal code that is understandable and self-documenting.
Use meaningful variable names. Avoid one-off helper functions.

After generating a first draft, revise your implementation with the following principles in mind:
- Minimize comments. Remove those that repeat what the code does. Keep comments which add meaningful context to _why_ or _how_ code behaviour works, but consider reducing the word count and simplifying to just one or two lines where possible.
- Human readability. A human will be reviewing and operating this codebase, so ensure it remains lightweight and interpretable. In your responses after making a change, explain what you accomplished and also _how_: what files were changed? explain the relevant diffs and give a brief walkthrough to ensure the operator remains in the loop.
- Feature precision. Do not implement new features that were not requested. Avoid making changes to files you are editing that are not required. Keep changes targeted to implenting only the requested feature (unless you are specifically asked to refactor or improve code quality, in which case you are free to make tweaks).
- Prefer direct runs and E2E evaluation over TDD. Use minimal unit testing: only for clear, challenging self-contained low-level units. Most behaviours do not require dedicated tests to evaluate their correctness: running once or twice and checking the output is often sufficient for most changes.
- Ask for clarification eagerly. Align with the user on scope, strategy, feature set, and potential roadblocks early.