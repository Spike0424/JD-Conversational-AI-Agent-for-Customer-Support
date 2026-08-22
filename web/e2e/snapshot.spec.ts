import { test, type Route } from '@playwright/test'

test('snapshot streaming with broken markdown', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 })
  await page.route('**/v1/auth/**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ access_token: 'mock.jwt.token' }) })
  )
  await page.route('**/v1/shops', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ id: 1001, shop_name: '测试旗舰店' }]) })
  )
  await page.route('**/v1/products**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ goods_id: 9001, goods_name: '智能保温杯', price: 199 }]) })
  )
  // The user's actual real LLM output (showing fragmentation artifacts)
  await page.route('**/v1/chat/stream', async (route: Route) => {
    const sse = 'data: ' + JSON.stringify({ delta: '小主您好，iPhone17的优势非常突出，为您梳理如下：\n1. 性能与屏幕\n-搭载苹果A19芯片，六核CPU+五核GPU，配合iOS26系统，日常使用极致流畅-6.\n3英寸OLED屏，120Hz高刷，460ppi，户外峰值亮度可达3000nits\n2. 影像与续航\n-后置4800万像素主摄，f/1.\n6大光圈，支持2倍光学变焦-视频播放最长30小时，支持40W有线快充+15WMagSafe无线充电\n3. 设计与防护\n-铝金属边框+超瓷晶面板，仅177g，手感轻盈-IP68防尘防水，支持双卡双待、蓝牙6.\n0和Wi-Fi7\n4. 购机保障\n-全新国行正品，全国联保1年，支持官方验机-下单即送配件礼包+碎屏险+延保，17点前下单当天顺丰发出这款256GB版仅需￥5999，性价比很高。\n库存紧俏，建议尽早下单哦😊' }) + '\n\ndata: [DONE]\n\n'
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
  await page.waitForTimeout(500)

  await page.locator('.ant-select-selector').first().click()
  await page.locator('.ant-select-item-option-content', { hasText: '测试旗舰店' }).click()
  await page.locator('.ant-select-selector').nth(1).click()
  await page.locator('.ant-select-item-option-content', { hasText: '智能保温杯' }).click()
  await page.getByRole('button', { name: '开始咨询' }).click()

  await page.getByPlaceholder(/输入您的问题/).fill('iPhone 17 优势')
  await page.getByPlaceholder(/输入您的问题/).press('Enter')
  await page.waitForTimeout(2000)

  // Captures both the rendered HTML in DOM and the visual screenshot
  const html = await page.evaluate(() => {
    return {
      text: document.querySelector('.ant-layout-content')?.textContent || 'NOT FOUND',
    }
  })
  console.log('=== Rendered text ===')
  console.log(html.text)

  await page.screenshot({ path: '/tmp/streaming-markdown.png', fullPage: false })
})
