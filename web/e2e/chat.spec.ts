import { test, expect, type Route } from '@playwright/test'

const JWT = 'mock.jwt.token'
const SHOP = '测试旗舰店'
const PRODUCT = '智能保温杯'

test('未登录首页 → 显示 demo 侧栏 + Hero 登录引导', async ({ page }) => {
  await page.goto('/')

  // 顶部 logo
  await expect(page.getByText('京东客服')).toBeVisible()
  // 开启新对话按钮
  await expect(page.getByRole('button', { name: /开启新对话/ })).toBeVisible()
  // 分日分组 + demo sessions
  await expect(page.getByText('今天').first()).toBeVisible()
  await expect(page.locator('.ant-layout-sider').getByText('智能保温杯')).toBeVisible()
  // 未登录 → Content 区显示 Hero 引导 + 大「登录 / 注册」按钮，不显示 ConsultationForm
  await expect(page.getByRole('heading', { name: '京东 AI 客服' })).toBeVisible()
  await expect(page.getByRole('button', { name: /登录\s*\/\s*注册/ }).first()).toBeVisible()
  await expect(page.getByRole('heading', { name: '新建对话' })).toHaveCount(0)
  // 顶部右侧「登录」按钮也还在
  await expect(page.getByRole('button', { name: '登录' }).first()).toBeVisible()
})

test('未登录 → 登录 → Modal 关闭', async ({ page }) => {
  await page.route('**/v1/auth/**', (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ access_token: JWT }),
    }),
  )

  await page.goto('/')
  await page.getByRole('button', { name: '登录' }).first().click()
  await page.locator('.ant-modal-content input[type="email"]').fill('test@example.com')
  await page.locator('.ant-modal-content input[type="password"]').fill('password123')
  await page.locator('.ant-modal-content button[type="submit"]').click()

  // Modal 关闭
  await expect(page.locator('.ant-modal-content')).toHaveCount(0, { timeout: 5_000 })
})

test('已登录 → 填表 → 进聊天 → 流式 chunks', async ({ page }) => {
  await page.route('**/v1/auth/**', (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ access_token: JWT }),
    }),
  )
  await page.route('**/v1/shops', (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([{ id: 1001, shop_name: SHOP }]),
    }),
  )
  await page.route('**/v1/products**', (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([{ goods_id: 9001, goods_name: PRODUCT, price: 199 }]),
    }),
  )
  await page.route('**/v1/chat/stream', async (route: Route) => {
    const sse = [
      'data: ' + JSON.stringify({ delta: '你好，' }) + '\n\n',
      'data: ' + JSON.stringify({ delta: '这款' }) + '\n\n',
      'data: ' + JSON.stringify({ delta: '商品' }) + '\n\n',
      'data: ' + JSON.stringify({ delta: '很好。' }) + '\n\n',
      'data: [DONE]\n\n',
    ].join('')
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' },
      body: sse,
    })
  })
  await page.route('**/v1/sessions', async (route: Route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ sessions: [] }),
      })
    } else {
      await route.fallback()
    }
  })

  // 登录
  await page.goto('/')
  await page.getByRole('button', { name: '登录' }).first().click()
  await page.locator('.ant-modal-content input[type="email"]').fill('test@example.com')
  await page.locator('.ant-modal-content input[type="password"]').fill('password123')
  await page.locator('.ant-modal-content button[type="submit"]').click()
  await expect(page.locator('.ant-modal-content')).toHaveCount(0, { timeout: 5_000 })

  // 填 ConsultationForm
  await page.locator('.ant-select-selector').first().click()
  await page.locator('.ant-select-item-option-content', { hasText: SHOP }).click()
  await page.locator('.ant-select-selector').nth(1).click()
  await page.locator('.ant-select-item-option-content', { hasText: PRODUCT }).click()
  await page.getByRole('button', { name: '开始咨询' }).click()

  // Header 显商品
  await expect(page.getByText(PRODUCT).first()).toBeVisible()

  // 发送消息（圆按钮无 name，靠 textarea 触发）
  await page.getByPlaceholder(/输入您的问题/).fill('有什么优势？')
  await page.getByPlaceholder(/输入您的问题/).press('Enter')

  await expect(page.getByText('你好，这款商品很好。')).toBeVisible({ timeout: 10_000 })
})
