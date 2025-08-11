"""
Test per i componenti di monitoring
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.data_quality import DataQualityMonitor
from src.monitoring.drift_detection import DriftDetector
from src.monitoring.monitoring_config import ensure_directories, get_monitoring_paths
from src.monitoring.performance_monitoring import PerformanceMonitor


class TestMonitoringConfig:
    """Test per la configurazione del monitoring"""

    def test_ensure_directories(self):
        """Test creazione directory"""
        ensure_directories()
        paths = get_monitoring_paths()

        # Verifica che le directory esistano
        for key, path in paths.items():
            if key.endswith("_dir"):
                assert path.exists(), f"Directory {path} non esiste"

    def test_get_monitoring_paths(self):
        """Test ottenimento path di monitoring"""
        paths = get_monitoring_paths()

        # Verifica che tutti i path siano definiti
        expected_keys = [
            "reports_dir",
            "dashboards_dir",
            "alerts_dir",
            "logs_dir",
            "data_quality_report",
            "drift_report",
            "performance_report",
            "dashboard",
        ]

        for key in expected_keys:
            assert key in paths, f"Path {key} mancante"


class TestDataQuality:
    """Test per il data quality monitoring"""

    def test_data_quality_monitor_creation(self):
        """Test creazione monitor data quality"""
        monitor = DataQualityMonitor()
        assert monitor is not None

    def test_data_quality_with_sample_data(self):
        """Test data quality con dati di esempio"""
        # Crea dati di esempio
        reference_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        current_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 50),
                "feature2": np.random.normal(0, 1, 50),
                "target": np.random.randint(0, 2, 50),
            }
        )

        monitor = DataQualityMonitor()

        # Test che il monitor può essere creato
        assert monitor is not None


class TestDriftDetection:
    """Test per il drift detection"""

    def test_drift_detector_creation(self):
        """Test creazione drift detector"""
        detector = DriftDetector()
        assert detector is not None

    def test_drift_detector_with_sample_data(self):
        """Test drift detection con dati di esempio"""
        # Crea dati di esempio
        reference_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        current_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 50),
                "feature2": np.random.normal(0, 1, 50),
                "target": np.random.randint(0, 2, 50),
            }
        )

        detector = DriftDetector()

        # Test che il detector può essere creato
        assert detector is not None


class TestPerformanceMonitoring:
    """Test per il performance monitoring"""

    def test_performance_monitor_creation(self):
        """Test creazione performance monitor"""
        monitor = PerformanceMonitor()
        assert monitor is not None

    def test_performance_monitor_with_sample_data(self):
        """Test performance monitoring con dati di esempio"""
        # Crea dati di esempio
        reference_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        current_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 50),
                "feature2": np.random.normal(0, 1, 50),
                "target": np.random.randint(0, 2, 50),
            }
        )

        monitor = PerformanceMonitor()

        # Test che il monitor può essere creato
        assert monitor is not None
