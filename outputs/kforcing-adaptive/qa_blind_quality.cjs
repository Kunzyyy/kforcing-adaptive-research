// Isolated, synthetic UI interactions only; no returned human rating files are created.
const {chromium}=require(process.env.BLIND_QA_PLAYWRIGHT || 'playwright');
const fs=require('node:fs'); const path=require('node:path'); const {pathToFileURL}=require('node:url'); const assert=require('node:assert/strict'); const crypto=require('node:crypto');
const root=path.join(__dirname,'results/blind-quality');
const dest=path.join(root,'ui-qa'); fs.mkdirSync(dest,{recursive:true});
const packet=JSON.parse(fs.readFileSync(path.join(root,'reviewer/tasks.json'),'utf8'));
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.BLIND_QA_CHROMIUM});
 const context=await browser.newContext({viewport:{width:1280,height:1000},acceptDownloads:true});
 const page=await context.newPage(); const errors=[];const remote=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))remote.push(r.url());});
 const url=pathToFileURL(path.join(root,'reviewer/index.html')).href; await page.goto(url);
 assert.match(await page.locator('#progressLabel').innerText(),/0 \/ 80/);
 assert.equal(await page.locator('#textA').innerText(),packet.tasks[0].A);
 assert.equal(await page.locator('input[name=choice]:checked').count(),0);
 await page.screenshot({path:path.join(dest,'desktop-blank.png'),fullPage:true});
 await page.locator('#export').click(); assert.match(await page.locator('#message').innerText(),/代号/);
 await page.locator('#alias').fill('synthetic-ui-test'); await page.locator('input[value=A]').check();
 await page.locator('#bothBad').check(); await page.locator('#comment').fill('Synthetic UI test only; not a human judgment.');
 await page.locator('#next').click();assert.match(await page.locator('#position').innerText(),/2/);
 assert.equal(await page.locator('input[name=choice]:checked').count(),0);
 await page.locator('input[value=SKIP]').check();await page.locator('#previous').click();assert.equal(await page.locator('input[value=A]').isChecked(),true);
 await page.reload();assert.equal(await page.locator('input[value=A]').isChecked(),true);assert.equal(await page.locator('#bothBad').isChecked(),true);
 assert.equal(await page.locator('#comment').inputValue(),'Synthetic UI test only; not a human judgment.');
 await page.locator('#unanswered').click();assert.equal(await page.locator('#jump').inputValue(),'2');
 const downloadPromise=page.waitForEvent('download');await page.locator('#export').click();const partialDownload=await downloadPromise;
 const partial=JSON.parse(fs.readFileSync(await partialDownload.path(),'utf8'));assert.equal(partial.complete,false);assert.equal(partial.answers.filter(a=>a.choice!==null).length,2);
 const mismatch={...partial,packet_id:'wrong-packet'};
 await page.locator('#import').setInputFiles({name:'wrong.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(mismatch))});
 await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('无法导入'));assert.match(await page.locator('#message').innerText(),/不属于/);
 // Reset isolated browser progress, then restore using the actual downloaded bytes.
 await page.evaluate(()=>localStorage.clear());await page.reload();assert.match(await page.locator('#progressLabel').innerText(),/0 \/ 80/);
 await page.locator('#import').setInputFiles({name:'progress.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(partial))});
 await page.waitForFunction(()=>document.querySelector('#message').textContent==='进度已导入。');assert.match(await page.locator('#progressLabel').innerText(),/2 \/ 80/);
 // Exercise all remaining UI controls; these machine-made selections never enter research ratings.
 for(let i=2;i<80;i++){await page.locator('#jump').selectOption(String(i));await page.locator('input[value=TIE]').check();}
 assert.match(await page.locator('#progressLabel').innerText(),/80 \/ 80/);
 await page.locator('#export').click();assert.match(await page.locator('#message').innerText(),/独立评审声明/);
 await page.locator('#human').check();const finalPromise=page.waitForEvent('download');await page.locator('#export').click();const finalDownload=await finalPromise;
 const final=JSON.parse(fs.readFileSync(await finalDownload.path(),'utf8'));assert.equal(final.complete,true);assert.equal(final.answers.length,80);assert.equal(final.reviewer.human_confirmed,true);
 assert.equal(new Set(final.answers.map(a=>a.task_id)).size,80);assert.equal(final.tasks_sha256,packet.tasks_sha256);
 // Clear all synthetic input before screenshots; browser context is also discarded at close.
 await page.evaluate(()=>localStorage.clear());await page.reload();await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
 await page.screenshot({path:path.join(dest,'mobile-blank.png'),fullPage:true});
 assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
 await browser.close();
 const result={passed:true,packet_id:packet.packet_id,script_sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),
   checks:['blank initial state','text fidelity','alias required','A/skip annotation and navigation','autosave/reload','unanswered navigation','partial JSON export','wrong-packet import rejected','partial restore','80 answers via UI','human declaration required','complete export schema','mobile no overflow','zero script errors','zero HTTP requests'],
   screenshots:['desktop-blank.png','mobile-blank.png'],synthetic_interactions_only:true,human_ratings_received:0,synthetic_rating_files_retained:false,isolated_browser_context_closed:true};
 fs.writeFileSync(path.join(dest,'checks.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
