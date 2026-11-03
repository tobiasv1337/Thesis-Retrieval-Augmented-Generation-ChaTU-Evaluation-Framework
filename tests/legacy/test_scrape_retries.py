import unittest
from unittest.mock import Mock, call, patch

import requests

from modules.scrape_az import _get_with_retries


class ScrapeRetryTests(unittest.TestCase):
    @patch("modules.scrape_az.time.sleep")
    @patch("modules.scrape_az.requests.get")
    def test_retries_three_times_and_reports_the_recovered_response(self, get, sleep):
        response = Mock(status_code=200)
        get.side_effect = [
            requests.ConnectionError("offline"),
            requests.ConnectTimeout("connect timeout"),
            requests.ReadTimeout("read timeout"),
            response,
        ]

        result = _get_with_retries("https://www.tu.berlin/example")

        self.assertIs(result, response)
        self.assertEqual(get.call_count, 4)
        self.assertEqual(sleep.call_args_list, [call(1), call(2), call(4)])

    @patch("modules.scrape_az.time.sleep")
    @patch("modules.scrape_az.requests.get")
    def test_raises_the_last_network_error_after_three_retries(self, get, sleep):
        get.side_effect = requests.ReadTimeout("read timeout")

        with self.assertRaises(requests.ReadTimeout):
            _get_with_retries("https://www.tu.berlin/example")

        self.assertEqual(get.call_count, 4)
        self.assertEqual(sleep.call_args_list, [call(1), call(2), call(4)])

    @patch("modules.scrape_az.time.sleep")
    @patch("modules.scrape_az.requests.get")
    def test_retries_transient_http_status_without_raising_for_final_status(self, get, sleep):
        unavailable = Mock(status_code=503)
        recovered = Mock(status_code=200)
        get.side_effect = [unavailable, unavailable, unavailable, recovered]

        result = _get_with_retries("https://www.tu.berlin/example")

        self.assertIs(result, recovered)
        self.assertEqual(get.call_count, 4)
        self.assertEqual(sleep.call_args_list, [call(1), call(2), call(4)])

        persistent = Mock(status_code=503)
        get.reset_mock()
        sleep.reset_mock()
        get.return_value = persistent
        get.side_effect = None

        result = _get_with_retries("https://www.tu.berlin/example")

        self.assertIs(result, persistent)
        self.assertEqual(get.call_count, 4)
        self.assertEqual(sleep.call_args_list, [call(1), call(2), call(4)])

    @patch("modules.scrape_az.time.sleep")
    @patch("modules.scrape_az.requests.get")
    def test_preserves_nonretryable_http_response(self, get, sleep):
        response = Mock(status_code=404)
        get.return_value = response

        result = _get_with_retries("https://www.tu.berlin/example")

        self.assertIs(result, response)
        get.assert_called_once_with(
            "https://www.tu.berlin/example",
            timeout=(3.05, 27),
        )
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
