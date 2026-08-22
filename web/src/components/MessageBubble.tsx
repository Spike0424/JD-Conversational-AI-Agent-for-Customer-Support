import ReactMarkdown from 'react-markdown'
import type { ChatMessage } from '../service/types'
import { ProductCardView } from './ProductCardView'

/**
 * Repair markdown that the LLM streamed in broken chunks.
 *
 * The model often emits `**1.` as one token and then `\n\n` and then
 * `**性能与屏幕**` as another. When the chunks are concatenated the
 * `**` markers get separated by `\n\n`, which breaks the bold span
 * (markdown closes a bold span at blank lines). We collapse internal
 * whitespace inside every `**...**` so the span stays intact.
 */
function normalizeMarkdown(text: string): string {
  // 0. Auto-promote numbered titles to bold even when the model forgot **.
  //    "1. 性能与屏幕" → "**1. 性能与屏幕**"
  //    Lookahead `[:：\-—\n]|$` prevents matching "6.3英寸" (decimal
  //    followed by Chinese). Title must start with a Chinese char. The
  //    optional "：|。\n" prefix allows matches mid-line after a colon or
  //    sentence ending — e.g. "梳理如下：1. 性能与屏幕".
  text = text.replace(
    /(?:^|[:：。\n])\s*(\d+\.)\s*([一-鿿][^\n]{0,30}?)(?=[:：\-—\n]|$)/g,
    (_m, num, title) => `**${num} ${title.trimEnd()}**`,
  )

  // 1. Collapse whitespace inside every closed bold span so "**1. \n\n性能与屏幕**"
  //    becomes "**1. 性能与屏幕**" — the bold span no longer gets split
  //    by a stray blank line.
  text = text.replace(/\*\*([^*]+?)\*\*(\s*)/g, (_m, content) =>
    `**${content.replace(/\s+/g, ' ').trim()}**`,
  )

  // 2. Collapse single \n between any non-\n chars. Folds streaming line
  //    breaks inside paragraphs (the model emits "每天\n17:00" as separate
  //    tokens).
  text = text.replace(/([^\n])\n(?=[^\n])/g, '$1')
  return text
}

export function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user'
  const renderedContent = isUser ? msg.content : normalizeMarkdown(msg.content)
  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 12,
    }}>
      <div style={{
        maxWidth: '70%',
        padding: '10px 14px',
        borderRadius: 16,
        background: isUser ? 'var(--color-primary)' : 'var(--bg-elevated)',
        color: 'var(--text-primary)',
        fontSize: 14,
        lineHeight: 1.6,
        wordBreak: 'break-word',
        whiteSpace: 'pre-wrap',
      }}>
        {isUser ? (
          renderedContent
        ) : (
          // `key` is incremented on each chunk so the ReactMarkdown instance
          // remounts and re-parses the AST. Without this, react-markdown v10
          // caches the parsed tree on length-equivalent content updates and
          // the UI shows raw `**bold**` text until the next page refresh.
          <ReactMarkdown
            key={renderedContent.length}
            components={{
              p: ({ children }) => <p style={{ margin: '0 0 10px 0', lineHeight: 1.7 }}>{children}</p>,
              strong: ({ children }) => (
                <strong
                  style={{
                    display: 'block',
                    fontWeight: 700,
                    fontSize: 15,
                    color: 'var(--color-primary)',
                    margin: '12px 0 6px 0',
                  }}
                >
                  {children}
                </strong>
              ),
              // Drop list semantics — render as inline text, no • markers.
              // The LLM's "- daily" / "17:00前" / "下单" sub-points all live
              // on a single line per section.
              ul: ({ children }) => <>{children}</>,
              ol: ({ children }) => <>{children}</>,
              li: ({ children }) => <span style={{ display: 'inline' }}>{children} </span>,
              em: ({ children }) => <em style={{ fontStyle: 'italic', color: 'var(--text-secondary)' }}>{children}</em>,
              a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--color-primary)' }}>{children}</a>,
              code: ({ children }) => <code style={{ background: 'var(--bg-canvas)', padding: '1px 4px', borderRadius: 3, fontFamily: 'monospace', fontSize: 13 }}>{children}</code>,
            }}
          >
            {renderedContent}
          </ReactMarkdown>
        )}
        {!isUser && 'product_cards' in msg && msg.product_cards?.map((c, i) => (
          <ProductCardView key={i} card={c} />
        ))}
      </div>
    </div>
  )
}
