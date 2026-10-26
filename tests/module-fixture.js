import vm from 'node:vm'
import fs from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

// Evaluate production modules with explicit external dependencies and no real credentials.
export async function loadModule(filename, mocks, globals = {}) {
    const context = vm.createContext({ console: { log() {}, warn() {}, error() {} },
        Buffer, URL, URLSearchParams, setTimeout, clearTimeout, ...globals })
    const cache = new Map()
    async function load(file) {
        if (cache.has(file)) return cache.get(file)
        const module = new vm.SourceTextModule(await fs.readFile(file, 'utf8'), {
            context, identifier: file,
            initializeImportMeta(meta) { meta.url = pathToFileURL(file).href }
        })
        cache.set(file, module)
        await module.link(async (specifier, parent) => {
            if (specifier === './database.js' && globals.testDatabase) {
                const exports = globals.testDatabase
                return new vm.SyntheticModule(Object.keys(exports), function () {
                    for (const [name, value] of Object.entries(exports)) this.setExport(name, value)
                }, { context })
            }
            if (specifier.startsWith('.')) return load(path.resolve(path.dirname(parent.identifier), specifier))
            const exports = Object.hasOwn(mocks, specifier) ? mocks[specifier] : await import(specifier)
            const names = Object.keys(exports)
            return new vm.SyntheticModule(names, function () {
                for (const name of names) this.setExport(name, exports[name])
            }, { context })
        })
        return module
    }
    const module = await load(path.resolve(filename))
    await module.evaluate()
    return module.namespace
}
