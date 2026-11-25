"""Every test gets its own settings file and no network.

PLASMACOOP_CONFIG points app_config at a file in the test's tmp_path, so no test reads Graham's settings or
writes over them (the event window saves its settings when it closes). And cdaweb's session factory fails the
test outright -- pytest.fail is not an Exception, so download() cannot swallow it -- unless the test installs
a fake server: a test that reached CDAWeb by accident would otherwise download for real.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _own_settings_and_no_network(tmp_path, monkeypatch):
	monkeypatch.setenv('PLASMACOOP_CONFIG', str(tmp_path / 'plasmacoop_config.json'))
	import cdaweb

	def no_network():
		pytest.fail('a test reached for CDAWeb without a fake server')

	monkeypatch.setattr(cdaweb, 'session_factory', no_network)
