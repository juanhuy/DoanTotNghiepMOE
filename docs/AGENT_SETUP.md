# Setup coding agent dùng chung

Root [AGENTS.md](../AGENTS.md) là nguồn chỉ dẫn. [WORKFLOWS.md](WORKFLOWS.md) giữ quy trình, [ARCHITECTURE.md](ARCHITECTURE.md) giữ bản đồ code. Không cần sao chép nhiều bộ persona/rules vào từng IDE.

## Codex

Repo có `.codex/config.toml`, hai role trong `.codex/agents/` và hai skill trong `.agents/skills/`. Khởi động Codex từ repo và kiểm tra `/skills`. Nếu project chưa trusted thì xem lại nội dung rồi trust theo UI của client; cấu hình project có thể không nạp trước khi trust. Phiên cũ có thể cần khởi động lại để khám phá skill/role mới.

- `moe-engine`: invariant engine/FFI, hồi quy và kiểm tra thích hợp.
- `moe-benchmark`: protocol hai máy và diễn giải bằng chứng.
- `reviewer`: review đọc-only theo chỉ dẫn.
- `evidence_reviewer`: audit provenance/claims đọc-only theo chỉ dẫn.

Model/reasoning/account/quyền hệ thống được kế thừa từ người dùng, không pin vào tài khoản cá nhân. Cấu hình cho phép tối đa ba thread; AGENTS chỉ dùng nhánh khi có ích. Ví dụ: “Sửa lỗi KV budget theo PLAN; dùng reviewer kiểm tra diff sau khi test đạt.” Không cần gọi mọi role cho một sửa nhỏ. Role không tạo sandbox đọc-only riêng: các subagent kế thừa quyền phiên cha.

Agent khác dùng cùng AGENTS/workflow và đọc hai SKILL.md như hướng dẫn khi client không auto-discover. Không tạo adapter config cho mọi IDE khi nhóm chưa dùng chúng. GitHub Copilot có thể được yêu cầu đọc AGENTS rõ ràng nếu client không tự nạp.

## MCP và plugins

Docs MCP chính thức được cấu hình **disabled/không bắt buộc** trong project. Chỉ bật khi cần tài liệu OpenAI/Codex; nó không cung cấp tài liệu Rust/CUDA và không cần để chạy engine. Khi bật, kiểm tra `codex mcp list` và thử truy vấn; file config tồn tại không có nghĩa đã handshake. Dùng shell/Git và primary docs cho việc local; không thêm filesystem/git/memory MCP trùng năng lực.

Đã tìm plugin GitHub trong Plugin Management: có tùy chọn kết nối repo/PR/issues, nhưng **chưa cài/chưa kết nối**. Hai người có thể dùng Git/PR bình thường; nếu sau này dùng connector, mỗi người tự cấp quyền repo cần thiết. Không commit token hoặc coi connection của một người là dùng được cho người kia. Không cài browser MCP trước khi có workflow E2E cụ thể.

## Các lựa chọn đã kiểm tra

| Nguồn | Quyết định |
| --- | --- |
| Skill discovery `vercel-labs/skills` | Dùng để nghiên cứu, bỏ khỏi dependency repo hằng ngày |
| `affaan-m/ecc` rust-patterns cũ | Thay bằng skill riêng ngắn, tập trung invariant thực tế; bỏ các quy tắc tuyệt đối/khuyến nghị dependency không dùng |
| `actionbook/rust-skills` | Đã xem GitHub README; bộ skill/agent/hook lớn hơn nhu cầu hai người, chưa cài |
| `openai/skills` | Tham khảo convention; không vendor các skill API/document/image không phục vụ runtime |
| Plugin squad tự chế | Bỏ manifest/persona không có bằng chứng client nạp; dùng native roles và workflow chung |

Hai skill hiện tại tự viết cho repo, version bằng Git; không cần `npx skills update` hoặc lockfile bên thứ ba. Không có post-install hook, auto-download hay thay cấu hình global. Kiểm tra frontmatter và TOML bằng `tools/check.py`; parser này chỉ xác nhận subset do repo sử dụng, không thay validator đầy đủ của client.

## Nguồn chính thức và nghiên cứu (đã tra 2026-10-22)

- [OpenAI: AGENTS.md](https://developers.openai.com/codex/guides/agents-md/) — chỉ dẫn project.
- [OpenAI: Skills](https://developers.openai.com/codex/skills/) — discovery `.agents/skills` và metadata.
- [OpenAI: Subagents](https://developers.openai.com/codex/subagents/) — role TOML và quyền kế thừa.
- [OpenAI: Config reference](https://developers.openai.com/codex/config-reference/) — config project/MCP.
- [OpenAI Docs MCP](https://developers.openai.com/resources/docs-mcp) — endpoint read-only.
- [GitHub: actionbook/rust-skills](https://github.com/actionbook/rust-skills) — ứng viên đã đánh giá, không dependency.
- [GitHub: openai/skills](https://github.com/openai/skills) — convention skill.

Khả năng sản phẩm có thể đổi theo phiên bản. CLI đang có trong phiên chuẩn hóa là 0.163.0; TOML hợp lệ không tự chứng minh role được chạy. CLI `codex mcp list --json` đã phát hiện Docs MCP trong project với enabled=false; chưa handshake vì server tắt. Chi tiết kiểm tra thực tế và giới hạn ghi trong PROGRESS.
