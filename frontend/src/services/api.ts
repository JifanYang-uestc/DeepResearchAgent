const baseURL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface ResearchRequest {
  topic: string;
  research_mode: ResearchMode;
  document_set_id?: string;
  search_api?: string;
}

export type ResearchMode = "web" | "document" | "hybrid";

export interface DocumentFileRecord {
  name: string;
  status: string;
  size: number;
  pages: number;
  notice?: string | null;
}

export interface DocumentSetResponse {
  document_set_id: string;
  status: "uploaded" | "indexing" | "ready" | "failed";
  documents: number;
  pages: number;
  chunks: number;
  files: DocumentFileRecord[];
  notices: string[];
}

export interface ResearchStreamEvent {
  type: string;
  [key: string]: unknown;
}

export interface StreamOptions {
  signal?: AbortSignal;
}

async function apiError(response: Response, fallback: string): Promise<Error> {
  const text = await response.text().catch(() => "");
  if (text) {
    try {
      const payload = JSON.parse(text) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        return new Error(payload.detail);
      }
    } catch {
      return new Error(text);
    }
  }
  return new Error(fallback);
}

export async function createDocumentSet(): Promise<DocumentSetResponse> {
  const response = await fetch(`${baseURL}/knowledge/document-sets`, {
    method: "POST"
  });
  if (!response.ok) {
    throw await apiError(response, "无法创建文档集");
  }
  return (await response.json()) as DocumentSetResponse;
}

export async function uploadDocumentSetFiles(
  documentSetId: string,
  files: File[]
): Promise<DocumentSetResponse> {
  const body = new FormData();
  for (const file of files) {
    body.append("files", file, file.name);
  }
  const response = await fetch(
    `${baseURL}/knowledge/document-sets/${documentSetId}/files`,
    { method: "POST", body }
  );
  if (!response.ok) {
    throw await apiError(response, "文档上传或索引构建失败");
  }
  return (await response.json()) as DocumentSetResponse;
}

export async function runResearchStream(
  payload: ResearchRequest,
  onEvent: (event: ResearchStreamEvent) => void,
  options: StreamOptions = {}
): Promise<void> {
  const response = await fetch(`${baseURL}/research/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream"
    },
    body: JSON.stringify(payload),
    signal: options.signal
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      errorText || `研究请求失败，状态码：${response.status}`
    );
  }

  const body = response.body;
  if (!body) {
    throw new Error("浏览器不支持流式响应，无法获取研究进度");
  }

  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);

      if (rawEvent.startsWith("data:")) {
        const dataPayload = rawEvent.slice(5).trim();
        if (dataPayload) {
          try {
            const event = JSON.parse(dataPayload) as ResearchStreamEvent;
            onEvent(event);

            if (event.type === "error" || event.type === "done") {
              return;
            }
          } catch (error) {
            console.error("解析流式事件失败：", error, dataPayload);
          }
        }
      }

      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      // 处理可能的尾巴事件
      if (buffer.trim()) {
        const rawEvent = buffer.trim();
        if (rawEvent.startsWith("data:")) {
          const dataPayload = rawEvent.slice(5).trim();
          if (dataPayload) {
            try {
              const event = JSON.parse(dataPayload) as ResearchStreamEvent;
              onEvent(event);
            } catch (error) {
              console.error("解析流式事件失败：", error, dataPayload);
            }
          }
        }
      }
      break;
    }
  }
}
