"""サーバー設定のテスト"""

from llasa_server.config import ServerConfig, get_config, set_config


class TestServerConfig:
    """ServerConfigデータクラスのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        config = ServerConfig()
        assert config.llasa_model_id == "NandemoGHS/Anime-Llasa-3B"
        assert config.xcodec2_model_id == "NandemoGHS/Anime-XCodec2"
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.8
        assert config.max_model_len == 2048
        assert config.device == "cuda"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.reload is False

    def test_custom_values(self):
        """カスタム値での初期化をテスト"""
        config = ServerConfig(
            llasa_model_id="custom/llasa",
            xcodec2_model_id="custom/xcodec2",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            device="cpu",
            host="127.0.0.1",
            port=9000,
            reload=True,
        )
        assert config.llasa_model_id == "custom/llasa"
        assert config.xcodec2_model_id == "custom/xcodec2"
        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.9
        assert config.max_model_len == 4096
        assert config.device == "cpu"
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.reload is True

    def test_partial_custom_values(self):
        """一部のみカスタム値での初期化をテスト"""
        config = ServerConfig(port=9000, device="cpu")
        assert config.port == 9000
        assert config.device == "cpu"
        # その他はデフォルト値
        assert config.llasa_model_id == "NandemoGHS/Anime-Llasa-3B"
        assert config.host == "0.0.0.0"


class TestConfigContextManagement:
    """設定のコンテキスト管理のテスト"""

    def test_get_config_returns_default(self):
        """get_config()がデフォルト設定を返すことをテスト"""
        # 新しいコンテキストでテストするため、設定をクリア
        config = get_config()
        assert isinstance(config, ServerConfig)
        assert config.port == 8000

    def test_set_and_get_config(self):
        """set_config()とget_config()の連携をテスト"""
        custom_config = ServerConfig(port=9000, host="127.0.0.1")
        set_config(custom_config)
        retrieved_config = get_config()
        assert retrieved_config.port == 9000
        assert retrieved_config.host == "127.0.0.1"

    def test_config_persistence(self):
        """設定の永続性をテスト"""
        custom_config = ServerConfig(
            port=7000, device="cpu", gpu_memory_utilization=0.5
        )
        set_config(custom_config)

        # 複数回取得しても同じ設定が返される
        config1 = get_config()
        config2 = get_config()
        assert config1.port == 7000
        assert config2.port == 7000
        assert config1.device == "cpu"
        assert config2.device == "cpu"

    def test_config_update(self):
        """設定の更新をテスト"""
        # 最初の設定
        config1 = ServerConfig(port=8000)
        set_config(config1)
        assert get_config().port == 8000

        # 設定を更新
        config2 = ServerConfig(port=9000)
        set_config(config2)
        assert get_config().port == 9000
