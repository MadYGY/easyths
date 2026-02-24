import functools

import mss


@functools.lru_cache(maxsize=1)
def get_mss_instance():
    """获取 mss 实例（全局单例）"""
    return mss.mss()
