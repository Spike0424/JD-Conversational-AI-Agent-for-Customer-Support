import { test, expect, type Route } from '@playwright/test'

const SHOP = '测试旗舰店'
const PRODUCT = '智能保温杯'

test('real LLM-shaped output (multiple sections on one line) renders cleanly', async ({ page }) => {
  await page.route('**/v1/auth/**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ access_token: 'mock.jwt.token' }) })
  )
  await page.route('**/v1/shops', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ id: 1001, shop_name: SHOP }]) })
  )
  await page.route('**/v1/products**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ goods_id: 9001, goods_name: PRODUCT, price: 199 }]) })
  )

  // EXACT shape from the user's screenshot — 3 sections on one line, comma-separated bullets
  await page.route('**/v1/chat/stream', async (route: Route) => {
    const realLLMOutput =
      '小主您好，iPhone17的优势非常突出，为您梳理如下：**1.性能与屏幕**-搭载苹果A19芯片，六核CPU+五核GPU，配合iOS26系统，日常使用极致流畅-6.3英寸OLED屏，120Hz高刷，460ppi，户外峰值亮度可达3000nits**2.影像与续航**-后置4800万像素主摄，f/1.6大光圈，支持2倍光学变焦-视频播放最长30小时，支持40W有线快充+15WMagSafe无线充电**3.设计与防护**-铝金属边框+超瓷晶面板，仅177g，手感轻盈-IP68防尘防水，支持双卡双待、蓝牙6.'
    const sse = 'data: ' + JSON.stringify({ delta: realLLMOutput }) + '\n\ndata: [DONE]\n\n'
    await route.fulfill({ status: 200, headers: { 'Content-Type': 'text/event-stream' }, body: sse })
  })
  await page.route('**/v1/sessions', (route: Route) =>
    route.request().method() === 'GET'
      ? route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sessions: [] }) })
      : route.fallback()
  )

  await page.goto('/')
  await page.getByRole('button', { name: '登录' }).first().click()
  await page.locator('.ant-modal-content input[type="email"]').fill('t@e.com')
  await page.locator('.ant-modal-content input[type="password"]').fill('p12345678')
  await page.locator('.ant-modal-content button[type="submit"]').click()
  await expect(page.locator('.ant-modal-content')).toHaveCount(0, { timeout: 5000 })

  await page.locator('.ant-select-selector').first().click()
  await page.locator('.ant-select-item-option-content', { hasText: SHOP }).click()
  await page.locator('.ant-select-selector').nth(1).click()
  await page.locator('.ant-select-item-option-content', { hasText: PRODUCT }).click()
  await page.getByRole('button', { name: '开始咨询' }).click()

  await page.getByPlaceholder(/输入您的问题/).fill('iPhone 17e 优势')
  await page.getByPlaceholder(/输入您的问题/).press('Enter')
  await expect(page.getByText('1.性能与屏幕')).toBeVisible({ timeout: 10_000 })
  await page.waitForTimeout(500)

  // Now: no frontend post-processing. Render LLM output as-is.
  // 3 inline bold titles appear (markdown *, **) — plus the ChatHeader title.
  const titles = await page.locator('strong').allTextContents()
  console.log('Bold titles:', titles)
  // Filter out the ChatHeader title (first "智能保温杯(199)") — keep only
  // assistant-bubble strongs.
  const assistantTitles = titles.filter((t) => /\d+\./.test(t))
  expect(assistantTitles.length).toBe(3)

  // No bullets (no broken list before the LLM output a single stream)
  const bullets = await page.locator('li').allTextContents()
  expect(bullets.length).toBe(0)

  // No special "heading" rendering — strong tag should be plain inline.
  const strongStyle = await page.locator('strong').first().evaluate((el) => {
    const cs = window.getComputedStyle(el)
    return { display: cs.display, color: cs.color, fontSize: cs.fontSize }
  })
  console.log('First strong style:', strongStyle)
  expect(strongStyle.display).toBe('inline')

  await page.screenshot({ path: '/home/spike/Workspace/Custom_answer_agent/pngs/after-fix.png', fullPage: false })
})
