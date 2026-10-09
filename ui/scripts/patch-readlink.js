const fs = require('fs');

function isFalsePositiveReadlinkError(err, pathValue) {
  if (!err || err.code !== 'EISDIR') {
    return false;
  }

  try {
    const stat = fs.lstatSync(pathValue);
    return !stat.isSymbolicLink();
  } catch {
    return false;
  }
}

function translateReadlinkError(err, pathValue) {
  if (!isFalsePositiveReadlinkError(err, pathValue)) {
    return err;
  }

  const translated = new Error(`EINVAL: invalid argument, readlink '${pathValue}'`);
  translated.code = 'EINVAL';
  translated.path = pathValue;
  translated.syscall = 'readlink';
  translated.cause = err;
  return translated;
}

function patchReadlinkModule(fsModule) {
  const originalReadlink = fsModule.readlink.bind(fsModule);
  const originalReadlinkSync = fsModule.readlinkSync.bind(fsModule);

  fsModule.readlink = function patchedReadlink(pathValue, options, callback) {
    if (typeof options === 'function') {
      callback = options;
      options = undefined;
    }

    return originalReadlink(pathValue, options, (err, linkString) => {
      if (callback) {
        callback(translateReadlinkError(err, pathValue), linkString);
      }
    });
  };

  fsModule.readlinkSync = function patchedReadlinkSync(pathValue, options) {
    try {
      return originalReadlinkSync(pathValue, options);
    } catch (err) {
      throw translateReadlinkError(err, pathValue);
    }
  };

  if (fsModule.promises && typeof fsModule.promises.readlink === 'function') {
    const originalPromisesReadlink = fsModule.promises.readlink.bind(fsModule.promises);
    fsModule.promises.readlink = async function patchedPromisesReadlink(pathValue, options) {
      try {
        return await originalPromisesReadlink(pathValue, options);
      } catch (err) {
        throw translateReadlinkError(err, pathValue);
      }
    };
  }
}

patchReadlinkModule(fs);
patchReadlinkModule(require('node:fs'));
