import json
from pathlib import Path

from odyssey.utils.directory_cache import directory_cache


def test_cached_directory_reuses_reads_and_invalidates_on_file_change(tmp_path):
    source = tmp_path / 'signal.json'
    source.write_text('{"value": 1}')
    calls = []

    @directory_cache
    def load(directory):
        calls.append(directory)
        return json.loads((Path(directory)/'signal.json').read_text())

    assert load(tmp_path) == {'value': 1}
    load(tmp_path)['value'] = -1
    assert load(tmp_path) == {'value': 1}
    assert len(calls) == 1
    source.write_text('{"value": 200}')
    assert load(tmp_path) == {'value': 200}
    assert len(calls) == 2


def test_failed_load_is_not_cached(tmp_path):
    import pytest
    @directory_cache
    def load(directory):
        return json.loads((Path(directory)/'a.json').read_text())
    (tmp_path/'a.json').write_text('bad')
    with pytest.raises(ValueError):load(tmp_path)
    (tmp_path/'a.json').write_text('{}')
    assert load(tmp_path)=={}
