const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '..', 'scripts',
  'vlm_screener', 'screener.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .split('/* ---------------- boot ---------------- */')[0];

function setup() {
  const elements = {};
  const speciesRadios = {rabbit: {checked: true}, pet: {checked: false}};
  const get = id => elements[id] ||= {
    value: '', innerHTML: '', textContent: '', scrollTop: 0,
    querySelectorAll(selector) {
      if (selector !== '.pager button') return [];
      this.buttons = [...this.innerHTML.matchAll(/<button data-pg="([^"]+)"/g)]
        .map(m => ({dataset: {pg: m[1]}}));
      return this.buttons;
    },
  };
  Object.assign(get('minN'), {value: '1'});
  Object.assign(get('minShare'), {value: '0'});
  Object.assign(get('fStatus'), {value: 'all'});
  Object.assign(get('fSource'), {value: 'all'});
  Object.assign(get('fSearch'), {value: ''});
  Object.assign(get('sort'), {value: 'name'});
  const context = vm.createContext({
    document: {
      getElementById: get, body: {},
      querySelector: selector => speciesRadios[selector.match(/value="(rabbit|pet)"/)[1]],
    }, window: {},
    fetch: () => new Promise(() => {}),
  });
  vm.runInContext(script, context);
  const videos = Array.from({length: 205}, (_, i) => ({
    vid: `v${String(i).padStart(3, '0')}`,
    name: `video-${i}`, rel_path: `video-${String(i).padStart(3, '0')}.mp4`,
    status: 'ready', duration: 3, n_segments: 1,
    combos: {'8b-v3': {dominant: 'eating', dominant_share: 1}},
  }));
  context.videos = videos;
  vm.runInContext('DATA={videos,n_videos:videos.length};curVid="v000";renderList()', context);
  return {context, elements, get, speciesRadios};
}

test('species selection changes combos and open request', () => {
  const {context, get, speciesRadios} = setup();
  vm.runInContext('applySpecies("pet")', context);
  assert.equal(speciesRadios.pet.checked, true);
  assert.match(get('inferCombos').innerHTML, /8b-v1/);
  assert.doesNotMatch(get('inferCombos').innerHTML, /8b-v3/);
  assert.equal(get('noneName').textContent, '無貓狗');
  get('sceneSelect').value = 'Site_A';
  assert.equal(JSON.parse(vm.runInContext('openPayload("C:/videos")', context)).species, 'pet');
  vm.runInContext('applySpecies("rabbit")', context);
  assert.match(get('inferCombos').innerHTML, /8b-v3/);
});

test('manual pagination is not overridden by the selected video', () => {
  const {context, elements, get} = setup();
  get('list').buttons.find(b => b.dataset.pg === 'next').onclick();
  assert.equal(vm.runInContext('page', context), 2);
  assert.match(get('list').innerHTML, /data-vid="v100"/);
  vm.runInContext('renderList()', context); // refresh should retain the page
  assert.equal(vm.runInContext('page', context), 2);
  vm.runInContext('showDetail(curVid,false)', context); // background detail refresh
  assert.equal(vm.runInContext('page', context), 2);
  get('list').buttons.find(b => b.dataset.pg === 'last').onclick();
  assert.equal(vm.runInContext('page', context), 3);
  get('pgJump').value = '2';
  get('pgJump').onchange();
  assert.equal(vm.runInContext('page', context), 2);
  elements.sort.onchange(); // filter/sort reset must not jump to curVid
  assert.equal(vm.runInContext('page', context), 1);
});

test('selecting a video across page boundaries still locates its page', () => {
  const {context, get} = setup();
  vm.runInContext('showDetail("v101")', context);
  assert.equal(vm.runInContext('page', context), 2);
  assert.match(get('list').innerHTML, /card sel" data-vid="v101"/);
  vm.runInContext('showDetail("v099")', context);
  assert.equal(vm.runInContext('page', context), 1);
});
