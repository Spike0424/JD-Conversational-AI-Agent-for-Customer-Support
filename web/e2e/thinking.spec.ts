import { test, expect, type Route } from '@playwright/test'

const SHOP = '测试旗舰店'
const PRODUCT = '智能保温杯'

test('thinking indicator shows before first SSE chunk', async ({ page }) => {
  await page.route('**/v1/auth/**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ access_token: 'mock.jwt.token' }) })
  )
  await page.route('**/v1/shops', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ id: 1001, shop_name: SHOP }]) })
  )
  await page.route('**/v1/products**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ goods_id: 9001, goods_name: PRODUCT, price: 199 }]) })
  )

  // Slow stream — first chunk delayed 200ms so we can catch the thinking state
  await page.route('**/v1/chat/stream', async (route: Route) => {
    const chunks = [
      'data: ' + JSON.stringify({ delta: '你好' }) + '\n\n',
      'data: ' + JSON.stringify({ delta: '世界' }) + '\n\n',
      'data: [DONE]\n\n',
    ]
    await new Promise((r) => setTimeout(r, 200))
    await route.fulfill({ status: 200, headers: { 'Content-Type': 'text/event-stream' }, body: chunks.join('') })
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

  await page.getByPlaceholder(/输入您的问题/).fill('hello')
  await page.getByPlaceholder(/输入您的问题/).press('Enter')

  // Within 100ms (before first chunk arrives at 200ms), thinking should show
  await expect(page.getByText('正在思考…')).toBeVisible({ timeout: 500 })

  // After chunks arrive, thinking disappears
  await expect(page.getByText('你好世界')).toBeVisible({ timeout: 5000 })
  await expect(page.getByText('正在思考…')).toHaveCount(0, { timeout: 5000 })
})