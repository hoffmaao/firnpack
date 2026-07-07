"""Low-level I/O for ApRES .DAT/.dat files.

This module is a Python translation of the MATLAB ApRES loaders in
`Codes_pRES_preprocess`, notably:

- LoadBurstRMB5_or.m
- LoadBurstRMB4.m
- LoadBurstRMB3.m
- func_file_format.m

It is intentionally conservative:
- Handles the common *one-burst-per-file* case used for GA stations, but
  also supports selecting `burst` from multi-burst files.
- Produces a compact `ApRESBurst` object containing the time-domain IF
  voltages (vif) and the radar parameters needed downstream.

If you prefer, you can switch to ImpDAR's loader by setting
`cfg.use_impdar_reader = True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import re

import numpy as np


# ----------------------------
# Dataclasses
# ----------------------------


@dataclass
class ApRESBurst:
    """A single ApRES burst (time-domain IF) plus minimal metadata."""

    file_path: Path
    file_format: int
    burst: int

    # Raw concatenated voltage record (Volts). Kept mainly for debugging.
    v: np.ndarray  # (words_per_burst,)

    # IF data matrix (Volts), one chirp per row
    vif: np.ndarray  # (nchirps, samples_per_chirp)

    # Burst geometry
    chirps_in_burst: int
    samples_per_chirp: int
    n_adc_samples: int
    subbursts_in_burst: int
    average: int

    # Attenuator settings (from header) and per-chirp attenuator code
    attenuator_1_db: np.ndarray  # (n_atten,)
    attenuator_2_db: np.ndarray  # (n_atten,)
    chirp_num: np.ndarray  # (nchirps,) 1-indexed
    chirp_att: np.ndarray  # (nchirps,) complex code (att1 + 1j*att2)

    # Minimal environmental metadata
    timestamp: datetime
    temperature_1_c: Optional[float]
    temperature_2_c: Optional[float]
    battery_voltage_v: Optional[float]

    # Optional GPS/position fields (often missing or 0 when GPS is off)
    gps_on: Optional[int]
    latitude_deg: Optional[float]
    longitude_deg: Optional[float]

    # FMCW radar parameters
    fs_hz: float
    f0_hz: float
    K_rad_s2: float
    er_ice: float

    # Derived parameters (Brennan nomenclature)
    dt_s: float
    T_s: float
    f1_hz: float
    B_hz: float
    fc_hz: float
    ci_m_s: float
    lambdac_m: float

    # Header parsing status
    code: int = 0

    @property
    def name(self) -> str:
        return self.file_path.stem


# ----------------------------
# File discovery
# ----------------------------


def find_dat_files(root: Path) -> List[Path]:
    """Recursively find ApRES raw data files under *root*.

    Includes both *.DAT and *.dat and ignores macOS AppleDouble files (._*).
    """

    root = Path(root)
    files: List[Path] = []
    for pat in ("*.DAT", "*.dat"):
        files.extend([p for p in root.rglob(pat) if p.is_file()])
    files = [p for p in files if not p.name.startswith("._")]
    return sorted(files)


# ----------------------------
# Small parsing helpers
# ----------------------------


def fmcw_file_format(path: Path) -> int:
    """Determine file format (MATLAB func_file_format).

    Returns
    -------
    int
        5 for RMB2/RMB5 (SW_Issue=...), 4/3 for older RMB files.
    """

    path = Path(path)
    with path.open("rb") as f:
        header = f.read(700)
    txt = header.decode("ascii", errors="ignore")

    if "SW_Issue=" in txt:
        return 5
    if "SubBursts in burst:" in txt:
        return 4
    if "*** Burst Header ***" in txt:
        return 3
    if "RADAR TIME" in txt:
        return 2

    raise ValueError(f"Unknown ApRES file format for {path}")


def _find_line_value(txt: str, key: str) -> Optional[str]:
    """Return the substring after *key* up to end-of-line."""

    idx = txt.find(key)
    if idx < 0:
        return None
    start = idx + len(key)
    end = txt.find("\n", start)
    if end < 0:
        end = len(txt)
    return txt[start:end].strip().rstrip("\r")


def _safe_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _parse_float_list(s: str) -> List[float]:
    # Accept comma/semicolon separated
    parts = s.replace(";", ",").split(",")
    out: List[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            continue
    return out


def _parse_int_list(s: str) -> List[int]:
    parts = s.replace(";", ",").split(",")
    out: List[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(float(p)))
        except Exception:
            continue
    return out


def _parse_timestamp(ts: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM:SS' into a timezone-aware datetime (UTC)."""

    dt = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc)


def read_dat_header_text(path: Path, *, max_bytes: int = 20_000) -> str:
    """Read and return the ASCII header block from an ApRES .DAT file.

    The RMB* formats used here store a plain-text header delimited by:
      "*** Burst Header ***"  ...  "*** End Header ***"

    We only read the first `max_bytes` bytes (fast) and truncate at the end
    marker when present.
    """

    path = Path(path)
    with path.open("rb") as f:
        b = f.read(int(max_bytes))
    end_marker = b"*** End Header ***"
    iend = b.find(end_marker)
    if iend != -1:
        b = b[: iend + len(end_marker)]
    return b.decode("ascii", errors="ignore")


def _parse_gps_fields(header_txt: str) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Parse GPS / position fields from a burst header.

    Many deployments have GPS disabled; in that case Latitude/Longitude are
    often present but set to 0.
    """

    gps_on = _safe_int(_find_line_value(header_txt, "GPSon="))
    lat = _safe_float(_find_line_value(header_txt, "Latitude="))
    lon = _safe_float(_find_line_value(header_txt, "Longitude="))

    # Regex fallbacks for headers with spaces around '='
    if gps_on is None:
        m = re.search(r"\bGPSon\s*=\s*(\d+)", header_txt, flags=re.IGNORECASE)
        if m:
            try:
                gps_on = int(m.group(1))
            except Exception:
                gps_on = None
    if lat is None:
        m = re.search(r"\bLatitude\s*=\s*([-+]?\d+(?:\.\d+)?)", header_txt, flags=re.IGNORECASE)
        if m:
            lat = _safe_float(m.group(1))
    if lon is None:
        m = re.search(r"\bLongitude\s*=\s*([-+]?\d+(?:\.\d+)?)", header_txt, flags=re.IGNORECASE)
        if m:
            lon = _safe_float(m.group(1))

    # Treat 0/0 as missing (common when GPS is off)
    if lat is not None and abs(lat) < 1e-9:
        lat = None
    if lon is not None and abs(lon) < 1e-9:
        lon = None

    # Sanity bounds
    if lat is not None and not (-90.0 <= lat <= 90.0):
        lat = None
    if lon is not None and not (-180.0 <= lon <= 180.0):
        lon = None

    return gps_on, lat, lon


def extract_lat_lon_from_dat(path: Path) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Fast helper to extract (lat, lon, gps_on) from a .DAT header."""

    hdr = read_dat_header_text(path)
    gps_on, lat, lon = _parse_gps_fields(hdr)
    return lat, lon, gps_on


def _derive_parameters(
    *,
    samples_per_chirp: int,
    fs_hz: float,
    f0_hz: float,
    K_rad_s2: float,
    er_ice: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """Translation of func_derive_parameters.m."""

    dt_s = 1.0 / fs_hz
    T_s = (samples_per_chirp - 1) / fs_hz
    f1_hz = f0_hz + T_s * K_rad_s2 / (2.0 * np.pi)
    B_hz = (samples_per_chirp / fs_hz) * (K_rad_s2 / (2.0 * np.pi))
    fc_hz = 0.5 * (f0_hz + f1_hz)
    ci_m_s = 299_792_458.0 / np.sqrt(er_ice)
    lambdac_m = ci_m_s / fc_hz
    return dt_s, T_s, f1_hz, B_hz, fc_hz, ci_m_s, lambdac_m


def _stable_unique(values: np.ndarray) -> np.ndarray:
    """Stable unique (like MATLAB unique(...,'stable'))."""

    seen = set()
    out = []
    for v in values.tolist():
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return np.asarray(out)


# ----------------------------
# RMB5 loader (format 5)
# ----------------------------


def load_burst_rmb5(
    filename: Path,
    burst: int,
    samples_per_chirp: int,
    *,
    fs_hz: float = 40000.0,
    f0_hz: float = 2.0e8,
    K_rad_s2: float = 2.0e8 * 2.0 * np.pi,
    er_ice: float = 3.18,
    max_header_len: int = 1500,
) -> ApRESBurst:
    """Load a burst from an RMB2/RMB5 ApRES file (LoadBurstRMB5_or.m)."""

    filename = Path(filename)

    with filename.open("rb") as f:
        f.seek(0, 2)
        filelength = f.tell()

        burstpointer = 0
        burstcount = 1
        header_txt = None

        # Variables for desired burst
        data_start = None
        w_per_chirp_cycle = None
        chirps_in_burst = None
        subbursts = None
        avg = None
        n_atten = None
        att1: List[float] = []
        att2: List[float] = []

        while burstcount <= burst and burstpointer <= filelength - max_header_len:
            f.seek(burstpointer, 0)
            header_bytes = f.read(max_header_len)
            header_txt = header_bytes.decode("ascii", errors="ignore")

            n_adc = _safe_int(_find_line_value(header_txt, "N_ADC_SAMPLES="))
            if n_adc is None:
                raise ValueError(f"Could not parse N_ADC_SAMPLES in {filename}")
            w_per_chirp_cycle = int(n_adc)

            subbursts = _safe_int(_find_line_value(header_txt, "NSubBursts="))
            if subbursts is None:
                raise ValueError(f"Could not parse NSubBursts in {filename}")

            avg = _safe_int(_find_line_value(header_txt, "Average="))
            if avg is None:
                avg = 0

            n_atten = _safe_int(_find_line_value(header_txt, "nAttenuators="))
            if n_atten is None:
                n_atten = 1

            att1_line = _find_line_value(header_txt, "Attenuator1=")
            att2_line = _find_line_value(header_txt, "AFGain=")
            att1 = (_parse_float_list(att1_line or "") + [0.0] * n_atten)[:n_atten]
            att2 = (_parse_float_list(att2_line or "") + [0.0] * n_atten)[:n_atten]

            tx_line = _find_line_value(header_txt, "TxAnt=") or ""
            rx_line = _find_line_value(header_txt, "RxAnt=") or ""
            tx = [x for x in _parse_int_list(tx_line) if x == 1]
            rx = [x for x in _parse_int_list(rx_line) if x == 1]
            if len(tx) == 0:
                tx = [1]
            if len(rx) == 0:
                rx = [1]

            if avg:
                chirps_in_burst = 1
            else:
                chirps_in_burst = int(subbursts * len(tx) * len(rx) * n_atten)

            end_key = "*** End Header ***"
            end_idx = header_txt.find(end_key)
            if end_idx < 0:
                raise ValueError(f"Could not find end header marker in {filename}")

            data_start = burstpointer + end_idx + len(end_key)
            words_per_burst = int(chirps_in_burst * w_per_chirp_cycle)
            bytes_per_word = 4 if avg == 2 else 2

            if burstcount < burst:
                burstpointer = data_start + words_per_burst * bytes_per_word
            else:
                burstpointer = data_start

            burstcount += 1

        if header_txt is None or data_start is None or w_per_chirp_cycle is None or chirps_in_burst is None:
            raise ValueError(f"Failed to locate burst {burst} in {filename}")

        # Parse remaining fields from header
        ts_line = _find_line_value(header_txt, "Time stamp=")
        if ts_line is None:
            raise ValueError(f"Could not parse Time stamp in {filename}")
        timestamp = _parse_timestamp(ts_line)

        t1 = _safe_float(_find_line_value(header_txt, "Temp1="))
        t2 = _safe_float(_find_line_value(header_txt, "Temp2="))
        bat = _safe_float(_find_line_value(header_txt, "BatteryVoltage="))
        if bat is None:
            bat = _safe_float(_find_line_value(header_txt, "BatteryVolt="))

        # Optional GPS/position fields
        gps_on, lat_deg, lon_deg = _parse_gps_fields(header_txt)

        # Override some radar parameters from header if present
        er_hdr = _safe_float(_find_line_value(header_txt, "ER_ICE="))
        if er_hdr is not None and er_hdr > 0:
            er_ice = float(er_hdr)

        # Sampling step can appear as TStepUp in seconds
        tstep_hdr = _safe_float(_find_line_value(header_txt, "TStepUp="))
        if tstep_hdr is not None and tstep_hdr > 0:
            fs_hz = float(1.0 / tstep_hdr)

        f0_hdr = _safe_float(_find_line_value(header_txt, "StartFreq="))
        f1_hdr = _safe_float(_find_line_value(header_txt, "StopFreq="))
        if f0_hdr is not None:
            f0_hz = float(f0_hdr)
        if f0_hdr is not None and f1_hdr is not None and fs_hz > 0:
            bandwidth_hz = float(f1_hdr - f0_hdr)
            K_rad_s2 = float(2.0 * np.pi * bandwidth_hz * fs_hz / samples_per_chirp)

        # Read burst voltage words
        f.seek(data_start, 0)
        words_per_burst = int(chirps_in_burst * w_per_chirp_cycle)
        if avg == 2:
            raw = np.fromfile(f, dtype="<u4", count=words_per_burst)
        else:
            raw = np.fromfile(f, dtype="<u2", count=words_per_burst)

    # Scale to volts (MATLAB: v * 2.5 / 2^16)
    v = raw.astype(np.float32) * (2.5 / (2**16))
    if avg == 2:
        v = v / (subbursts * n_atten)

    # Build vif matrix
    startind = np.arange(0, chirps_in_burst * w_per_chirp_cycle, w_per_chirp_cycle, dtype=int)
    vif = np.zeros((chirps_in_burst, samples_per_chirp), dtype=np.float32)
    for i in range(chirps_in_burst):
        s = startind[i]
        e = min(s + samples_per_chirp, v.size)
        if e - s == samples_per_chirp:
            vif[i, :] = v[s:e]
        else:
            tmp = np.zeros(samples_per_chirp, dtype=np.float32)
            if e > s:
                tmp[: e - s] = v[s:e]
            vif[i, :] = tmp

    # Derive radar params
    dt_s, T_s, f1_hz, B_hz, fc_hz, ci_m_s, lambdac_m = _derive_parameters(
        samples_per_chirp=samples_per_chirp,
        fs_hz=fs_hz,
        f0_hz=f0_hz,
        K_rad_s2=K_rad_s2,
        er_ice=er_ice,
    )

    # Build per-chirp attenuator code (cycles through attenuator list)
    att_set = np.asarray(att1, dtype=float) + 1j * np.asarray(att2, dtype=float)
    if att_set.size == 0:
        att_set = np.asarray([0.0 + 0.0j])
    chirp_num = np.arange(chirps_in_burst, dtype=int) + 1
    chirp_att = np.asarray([att_set[(k % att_set.size)] for k in range(chirps_in_burst)], dtype=np.complex128)

    return ApRESBurst(
        file_path=filename,
        file_format=5,
        burst=burst,
        v=v,
        vif=vif,
        chirps_in_burst=int(chirps_in_burst),
        samples_per_chirp=int(samples_per_chirp),
        n_adc_samples=int(w_per_chirp_cycle),
        subbursts_in_burst=int(subbursts),
        average=int(avg),
        attenuator_1_db=np.asarray(att1, dtype=float),
        attenuator_2_db=np.asarray(att2, dtype=float),
        chirp_num=chirp_num,
        chirp_att=chirp_att,
        timestamp=timestamp,
        temperature_1_c=t1,
        temperature_2_c=t2,
        battery_voltage_v=bat,
        gps_on=gps_on,
        latitude_deg=lat_deg,
        longitude_deg=lon_deg,
        fs_hz=float(fs_hz),
        f0_hz=float(f0_hz),
        K_rad_s2=float(K_rad_s2),
        er_ice=float(er_ice),
        dt_s=float(dt_s),
        T_s=float(T_s),
        f1_hz=float(f1_hz),
        B_hz=float(B_hz),
        fc_hz=float(fc_hz),
        ci_m_s=float(ci_m_s),
        lambdac_m=float(lambdac_m),
        code=0,
    )


# ----------------------------
# RMB4 loader (format 4)
# ----------------------------


def load_burst_rmb4(
    filename: Path,
    burst: int,
    samples_per_chirp: int,
    *,
    fs_hz: float = 40000.0,
    f0_hz: float = 2.0e8,
    K_rad_s2: float = 2.0e8 * 2.0 * np.pi,
    er_ice: float = 3.18,
    max_header_len: int = 1200,
) -> ApRESBurst:
    """Load a burst from RMB4-style files (LoadBurstRMB4.m)."""

    filename = Path(filename)

    with filename.open("rb") as f:
        f.seek(0, 2)
        filelength = f.tell()

        burstpointer = 0
        burstcount = 1
        header_txt = None

        data_start = None
        w_per_chirp_cycle = None
        chirps_in_burst = None
        subbursts = None
        avg = None
        n_atten = None
        att1: List[float] = []
        att2: List[float] = []

        while burstcount <= burst and burstpointer <= filelength - max_header_len:
            f.seek(burstpointer, 0)
            header_bytes = f.read(max_header_len)
            header_txt = header_bytes.decode("ascii", errors="ignore")

            n_samples = _safe_int(_find_line_value(header_txt, "Samples:"))
            if n_samples is None:
                raise ValueError(f"Could not parse 'Samples:' in {filename}")
            w_per_chirp_cycle = int(n_samples)

            subbursts = _safe_int(_find_line_value(header_txt, "SubBursts in burst:"))
            if subbursts is None:
                raise ValueError(f"Could not parse 'SubBursts in burst:' in {filename}")

            avg = _safe_int(_find_line_value(header_txt, "Average:"))
            if avg is None:
                avg = 0

            n_atten = _safe_int(_find_line_value(header_txt, "nAttenuators:"))
            if n_atten is None:
                n_atten = 1

            att1_line = _find_line_value(header_txt, "Attenuator 1:")
            att2_line = _find_line_value(header_txt, "Attenuator 2:")
            att1 = (_parse_float_list(att1_line or "") + [0.0] * n_atten)[:n_atten]
            att2 = (_parse_float_list(att2_line or "") + [0.0] * n_atten)[:n_atten]

            tx_line = _find_line_value(header_txt, "Tx Antenna select:") or ""
            rx_line = _find_line_value(header_txt, "Rx Antenna select:") or ""
            tx = [x for x in _parse_int_list(tx_line) if x == 1]
            rx = [x for x in _parse_int_list(rx_line) if x == 1]
            if len(tx) == 0:
                tx = [1]
            if len(rx) == 0:
                rx = [1]

            if avg:
                chirps_in_burst = 1
            else:
                chirps_in_burst = int(subbursts * len(tx) * len(rx) * n_atten)

            end_key = "*** End Header ***"
            end_idx = header_txt.find(end_key)
            if end_idx < 0:
                raise ValueError(f"Could not find end header marker in {filename}")

            data_start = burstpointer + end_idx + len(end_key)
            words_per_burst = int(chirps_in_burst * w_per_chirp_cycle)
            bytes_per_word = 4 if avg == 2 else 2

            if burstcount < burst:
                burstpointer = data_start + words_per_burst * bytes_per_word
            else:
                burstpointer = data_start

            burstcount += 1

        if header_txt is None or data_start is None or w_per_chirp_cycle is None or chirps_in_burst is None:
            raise ValueError(f"Failed to locate burst {burst} in {filename}")

        # Parse remaining fields
        ts_line = _find_line_value(header_txt, "Time stamp:")
        if ts_line is None:
            raise ValueError(f"Could not parse 'Time stamp:' in {filename}")
        timestamp = _parse_timestamp(ts_line)

        t1 = _safe_float(_find_line_value(header_txt, "Temperature 1:"))
        t2 = _safe_float(_find_line_value(header_txt, "Temperature 2:"))
        bat = _safe_float(_find_line_value(header_txt, "Battery voltage:"))

        # Optional GPS/position fields (may be absent in RMB3 headers)
        gps_on, lat_deg, lon_deg = _parse_gps_fields(header_txt)

        # Optional GPS/position fields (may be absent in RMB4 headers)
        gps_on, lat_deg, lon_deg = _parse_gps_fields(header_txt)

        # Read burst samples
        f.seek(data_start, 0)
        words_per_burst = int(chirps_in_burst * w_per_chirp_cycle)
        if avg == 2:
            raw = np.fromfile(f, dtype="<u4", count=words_per_burst)
        else:
            raw = np.fromfile(f, dtype="<u2", count=words_per_burst)

    v = raw.astype(np.float32) * (2.5 / (2**16))
    if avg == 2:
        v = v / (subbursts * n_atten)

    startind = np.arange(0, chirps_in_burst * w_per_chirp_cycle, w_per_chirp_cycle, dtype=int)
    vif = np.zeros((chirps_in_burst, samples_per_chirp), dtype=np.float32)
    for i in range(chirps_in_burst):
        s = startind[i]
        e = min(s + samples_per_chirp, v.size)
        if e - s == samples_per_chirp:
            vif[i, :] = v[s:e]
        else:
            tmp = np.zeros(samples_per_chirp, dtype=np.float32)
            if e > s:
                tmp[: e - s] = v[s:e]
            vif[i, :] = tmp

    dt_s, T_s, f1_hz, B_hz, fc_hz, ci_m_s, lambdac_m = _derive_parameters(
        samples_per_chirp=samples_per_chirp,
        fs_hz=fs_hz,
        f0_hz=f0_hz,
        K_rad_s2=K_rad_s2,
        er_ice=er_ice,
    )

    att_set = np.asarray(att1, dtype=float) + 1j * np.asarray(att2, dtype=float)
    if att_set.size == 0:
        att_set = np.asarray([0.0 + 0.0j])
    chirp_num = np.arange(chirps_in_burst, dtype=int) + 1
    chirp_att = np.asarray([att_set[(k % att_set.size)] for k in range(chirps_in_burst)], dtype=np.complex128)

    return ApRESBurst(
        file_path=filename,
        file_format=4,
        burst=burst,
        v=v,
        vif=vif,
        chirps_in_burst=int(chirps_in_burst),
        samples_per_chirp=int(samples_per_chirp),
        n_adc_samples=int(w_per_chirp_cycle),
        subbursts_in_burst=int(subbursts),
        average=int(avg),
        attenuator_1_db=np.asarray(att1, dtype=float),
        attenuator_2_db=np.asarray(att2, dtype=float),
        chirp_num=chirp_num,
        chirp_att=chirp_att,
        timestamp=timestamp,
        temperature_1_c=t1,
        temperature_2_c=t2,
        battery_voltage_v=bat,
        gps_on=gps_on,
        latitude_deg=lat_deg,
        longitude_deg=lon_deg,
        fs_hz=float(fs_hz),
        f0_hz=float(f0_hz),
        K_rad_s2=float(K_rad_s2),
        er_ice=float(er_ice),
        dt_s=float(dt_s),
        T_s=float(T_s),
        f1_hz=float(f1_hz),
        B_hz=float(B_hz),
        fc_hz=float(fc_hz),
        ci_m_s=float(ci_m_s),
        lambdac_m=float(lambdac_m),
        code=0,
    )


# ----------------------------
# RMB3 loader (format 3)
# ----------------------------


def load_burst_rmb3(
    filename: Path,
    burst: int,
    samples_per_chirp: int,
    *,
    fs_hz: float = 40000.0,
    f0_hz: float = 2.0e8,
    K_rad_s2: float = 2.0e8 * 2.0 * np.pi,
    er_ice: float = 3.18,
    max_header_len: int = 1200,
) -> ApRESBurst:
    """Load a burst from RMB3-style files (LoadBurstRMB3.m)."""

    filename = Path(filename)

    with filename.open("rb") as f:
        f.seek(0, 2)
        filelength = f.tell()

        burstpointer = 0
        burstcount = 1
        header_txt = None

        data_start = None
        w_per_chirp_cycle = None
        chirps_in_burst = None
        subbursts = 1
        avg = 0
        n_atten = 4
        att1: List[float] = []
        att2: List[float] = []

        while burstcount <= burst and burstpointer <= filelength - max_header_len:
            f.seek(burstpointer, 0)
            header_bytes = f.read(max_header_len)
            header_txt = header_bytes.decode("ascii", errors="ignore")

            n_samples = _safe_int(_find_line_value(header_txt, "Samples:"))
            if n_samples is None:
                raise ValueError(f"Could not parse 'Samples:' in {filename}")
            w_per_chirp_cycle = int(n_samples)

            chirps_in_burst = _safe_int(_find_line_value(header_txt, "Chirps in burst:"))
            if chirps_in_burst is None:
                raise ValueError(f"Could not parse 'Chirps in burst:' in {filename}")
            chirps_in_burst = int(chirps_in_burst)

            end_key = "*** End Header ***"
            end_idx = header_txt.find(end_key)
            if end_idx < 0:
                raise ValueError(f"Could not find end header marker in {filename}")

            data_start = burstpointer + end_idx + len(end_key)
            words_per_burst = int(chirps_in_burst * w_per_chirp_cycle)
            bytes_per_word = 2  # int16

            if burstcount < burst:
                burstpointer = data_start + words_per_burst * bytes_per_word
            else:
                burstpointer = data_start

            burstcount += 1

        if header_txt is None or data_start is None or w_per_chirp_cycle is None or chirps_in_burst is None:
            raise ValueError(f"Failed to locate burst {burst} in {filename}")

        ts_line = _find_line_value(header_txt, "Time stamp:")
        if ts_line is None:
            raise ValueError(f"Could not parse 'Time stamp:' in {filename}")
        timestamp = _parse_timestamp(ts_line)

        t1 = _safe_float(_find_line_value(header_txt, "Temperature 1:"))
        t2 = _safe_float(_find_line_value(header_txt, "Temperature 2:"))
        bat = _safe_float(_find_line_value(header_txt, "Battery voltage:"))

        att1_line = _find_line_value(header_txt, "Attenuator 1:")
        att2_line = _find_line_value(header_txt, "Attenuator 2:")
        if att1_line is not None:
            att1 = (_parse_float_list(att1_line) + [0.0] * n_atten)[:n_atten]
        else:
            att1 = [0.0] * n_atten
        if att2_line is not None:
            att2 = (_parse_float_list(att2_line) + [0.0] * n_atten)[:n_atten]
        else:
            att2 = [0.0] * n_atten

        # Read burst samples
        f.seek(data_start, 0)
        words_per_burst = int(chirps_in_burst * w_per_chirp_cycle)
        raw = np.fromfile(f, dtype="<i2", count=words_per_burst)

    # Scale signed int16 to 0..2.5 V (MATLAB: v*2.5/2^16 + 1.25)
    v = raw.astype(np.float32) * (2.5 / (2**16)) + 1.25

    startind = np.arange(0, chirps_in_burst * w_per_chirp_cycle, w_per_chirp_cycle, dtype=int)
    vif = np.zeros((chirps_in_burst, samples_per_chirp), dtype=np.float32)
    for i in range(chirps_in_burst):
        s = startind[i]
        e = min(s + samples_per_chirp, v.size)
        if e - s == samples_per_chirp:
            vif[i, :] = v[s:e]
        else:
            tmp = np.zeros(samples_per_chirp, dtype=np.float32)
            if e > s:
                tmp[: e - s] = v[s:e]
            vif[i, :] = tmp

    dt_s, T_s, f1_hz, B_hz, fc_hz, ci_m_s, lambdac_m = _derive_parameters(
        samples_per_chirp=samples_per_chirp,
        fs_hz=fs_hz,
        f0_hz=f0_hz,
        K_rad_s2=K_rad_s2,
        er_ice=er_ice,
    )

    att_set = np.asarray(att1, dtype=float) + 1j * np.asarray(att2, dtype=float)
    if att_set.size == 0:
        att_set = np.asarray([0.0 + 0.0j])
    chirp_num = np.arange(chirps_in_burst, dtype=int) + 1
    chirp_att = np.asarray([att_set[(k % att_set.size)] for k in range(chirps_in_burst)], dtype=np.complex128)

    return ApRESBurst(
        file_path=filename,
        file_format=3,
        burst=burst,
        v=v,
        vif=vif,
        chirps_in_burst=int(chirps_in_burst),
        samples_per_chirp=int(samples_per_chirp),
        n_adc_samples=int(w_per_chirp_cycle),
        subbursts_in_burst=int(subbursts),
        average=int(avg),
        attenuator_1_db=np.asarray(att1, dtype=float),
        attenuator_2_db=np.asarray(att2, dtype=float),
        chirp_num=chirp_num,
        chirp_att=chirp_att,
        timestamp=timestamp,
        temperature_1_c=t1,
        temperature_2_c=t2,
        battery_voltage_v=bat,
        gps_on=gps_on,
        latitude_deg=lat_deg,
        longitude_deg=lon_deg,
        fs_hz=float(fs_hz),
        f0_hz=float(f0_hz),
        K_rad_s2=float(K_rad_s2),
        er_ice=float(er_ice),
        dt_s=float(dt_s),
        T_s=float(T_s),
        f1_hz=float(f1_hz),
        B_hz=float(B_hz),
        fc_hz=float(fc_hz),
        ci_m_s=float(ci_m_s),
        lambdac_m=float(lambdac_m),
        code=0,
    )


# ----------------------------
# Public loader (optional ImpDAR)
# ----------------------------


def load_apres_burst(
    path: Path,
    *,
    burst: int = 1,
    samples_per_chirp: int = 40000,
    fs_hz: float = 40000.0,
    f0_hz: float = 2.0e8,
    K_rad_s2: float = 2.0e8 * 2.0 * np.pi,
    er_ice: float = 3.18,
    use_impdar: bool = False,
) -> ApRESBurst:
    """Load an ApRES burst from disk.

    Parameters
    ----------
    use_impdar
        If True, tries to use ImpDAR's ApRES reader. Otherwise uses the
        built-in readers translated from MATLAB.
    """

    path = Path(path)

    if use_impdar:
        try:
            from impdar.lib.ApresData.load_apres import load_apres as _impdar_load_apres  # type: ignore
        except Exception as exc:
            raise ImportError(
                "use_impdar=True but ImpDAR could not be imported. Install impdar or set use_impdar=False."
            ) from exc

        ad = _impdar_load_apres([str(path)])
        data = np.asarray(ad.data)
        if data.ndim == 3:
            vif = np.asarray(data[0], dtype=np.float32)
        elif data.ndim == 2:
            vif = np.asarray(data, dtype=np.float32)
        else:
            raise ValueError(f"Unexpected ImpDAR data shape {data.shape} for {path}")

        if vif.shape[1] > samples_per_chirp:
            vif = vif[:, :samples_per_chirp]
        elif vif.shape[1] < samples_per_chirp:
            raise ValueError(
                f"ImpDAR returned {vif.shape[1]} samples/chirp but samples_per_chirp={samples_per_chirp}."
            )

        # Best-effort timestamps/params
        ts = None
        if hasattr(ad, "time_stamp"):
            try:
                ts = ad.time_stamp[0]
            except Exception:
                ts = None
        if ts is None:
            ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Try to grab header fields
        try:
            hdr = getattr(ad, "header")
            fs_hz = float(getattr(hdr, "fs", fs_hz))
            f0_hz = float(getattr(hdr, "f0", f0_hz))
            K_rad_s2 = float(getattr(hdr, "chirp_grad", K_rad_s2))
            er_ice = float(getattr(hdr, "er", er_ice))
        except Exception:
            pass

        dt_s, T_s, f1_hz, B_hz, fc_hz, ci_m_s, lambdac_m = _derive_parameters(
            samples_per_chirp=samples_per_chirp,
            fs_hz=fs_hz,
            f0_hz=f0_hz,
            K_rad_s2=K_rad_s2,
            er_ice=er_ice,
        )

        chirps_in_burst = int(vif.shape[0])
        chirp_num = np.arange(chirps_in_burst, dtype=int) + 1
        chirp_att = np.zeros(chirps_in_burst, dtype=np.complex128)

        # Best-effort position fields from the raw file header
        gps_on, lat_deg, lon_deg = None, None, None
        try:
            hdr_txt = read_dat_header_text(path)
            gps_on, lat_deg, lon_deg = _parse_gps_fields(hdr_txt)
        except Exception:
            pass

        return ApRESBurst(
            file_path=path,
            file_format=5,
            burst=burst,
            v=np.asarray([], dtype=np.float32),
            vif=vif,
            chirps_in_burst=chirps_in_burst,
            samples_per_chirp=int(samples_per_chirp),
            n_adc_samples=int(samples_per_chirp),
            subbursts_in_burst=chirps_in_burst,
            average=0,
            attenuator_1_db=np.asarray([0.0], dtype=float),
            attenuator_2_db=np.asarray([0.0], dtype=float),
            chirp_num=chirp_num,
            chirp_att=chirp_att,
            timestamp=ts,
            temperature_1_c=None,
            temperature_2_c=None,
            battery_voltage_v=None,
            gps_on=gps_on,
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            fs_hz=float(fs_hz),
            f0_hz=float(f0_hz),
            K_rad_s2=float(K_rad_s2),
            er_ice=float(er_ice),
            dt_s=float(dt_s),
            T_s=float(T_s),
            f1_hz=float(f1_hz),
            B_hz=float(B_hz),
            fc_hz=float(fc_hz),
            ci_m_s=float(ci_m_s),
            lambdac_m=float(lambdac_m),
            code=0,
        )

    # Built-in readers
    fmt = fmcw_file_format(path)
    if fmt == 5:
        return load_burst_rmb5(
            filename=path,
            burst=burst,
            samples_per_chirp=samples_per_chirp,
            fs_hz=fs_hz,
            f0_hz=f0_hz,
            K_rad_s2=K_rad_s2,
            er_ice=er_ice,
        )
    if fmt == 4:
        return load_burst_rmb4(
            filename=path,
            burst=burst,
            samples_per_chirp=samples_per_chirp,
            fs_hz=fs_hz,
            f0_hz=f0_hz,
            K_rad_s2=K_rad_s2,
            er_ice=er_ice,
        )
    if fmt == 3:
        return load_burst_rmb3(
            filename=path,
            burst=burst,
            samples_per_chirp=samples_per_chirp,
            fs_hz=fs_hz,
            f0_hz=f0_hz,
            K_rad_s2=K_rad_s2,
            er_ice=er_ice,
        )

    raise NotImplementedError(f"File format {fmt} is not implemented for {path}")
