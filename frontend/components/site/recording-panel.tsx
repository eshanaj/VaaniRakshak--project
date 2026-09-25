'use client';

import { useState, useEffect, useRef } from 'react';

import { Mic, Square } from 'lucide-react';

import { WaveformVisualizer } from '@/components/site/waveform-visualizer';

import { cn } from '@/lib/utils';

interface RecordingPanelProps {
  onRecordingComplete: (file: File, duration: number) => void;
  disabled?: boolean;
}

export function RecordingPanel({
  onRecordingComplete,
  disabled,
}: RecordingPanelProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setElapsed((prev) => prev + 1);
      }, 1000);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isRecording]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;

    return `${m.toString().padStart(2, '0')}:${sec
      .toString()
      .padStart(2, '0')}`;
  };

  const handleStart = async () => {
    if (disabled || isRecording) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      streamRef.current = stream;

      let mimeType = '';

      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus';
      } else if (MediaRecorder.isTypeSupported('audio/webm')) {
        mimeType = 'audio/webm';
      } else {
        throw new Error(
          'This browser does not support WebM audio recording.'
        );
      }

      const mediaRecorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined
      );

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
      };

      mediaRecorder.start(250);

      setElapsed(0);
      setIsRecording(true);
    } catch (error) {
      console.error('Microphone recording error:', error);

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      alert(
        'Unable to start microphone recording. Please check your microphone permission and try again.'
      );
    }
  };

  const handleStop = () => {
    const mediaRecorder = mediaRecorderRef.current;

    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      return;
    }

    mediaRecorder.onstop = () => {
      try {
        const finalMimeType =
          mediaRecorder.mimeType || 'audio/webm;codecs=opus';

        const audioBlob = new Blob(audioChunksRef.current, {
          type: finalMimeType,
        });

        if (audioBlob.size === 0) {
          console.error('Recorded audio is empty.');
          alert('No audio was recorded. Please try again.');
          return;
        }

        const recordedFile = new File(
          [audioBlob],
          `recording_${Date.now()}.webm`,
          {
            type: finalMimeType,
          }
        );

        // TEMPORARY DEBUG:
        // Download a copy of the actual microphone recording.
        const downloadUrl = URL.createObjectURL(audioBlob);
        const downloadLink = document.createElement('a');
        downloadLink.href = downloadUrl;
        downloadLink.download = recordedFile.name;
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
        URL.revokeObjectURL(downloadUrl);

        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
        }

        mediaRecorderRef.current = null;
        audioChunksRef.current = [];

        setIsRecording(false);

        onRecordingComplete(recordedFile, elapsed);
      } catch (error) {
        console.error('Failed to create recording file:', error);

        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
        }

        setIsRecording(false);
      }
    };

    mediaRecorder.stop();
  };

  return (
    <div className="flex flex-col items-center gap-5 rounded-xl border border-cyan-500/15 bg-cyan-500/5 p-6">
      <div className="relative flex items-center justify-center">
        {isRecording && (
          <>
            <div className="absolute h-20 w-20 rounded-full border-2 border-cyan-400/40 animate-pulse-ring" />

            <div
              className="absolute h-20 w-20 rounded-full border-2 border-cyan-400/30 animate-pulse-ring"
              style={{ animationDelay: '0.7s' }}
            />
          </>
        )}

        <button
          onClick={isRecording ? handleStop : handleStart}
          disabled={disabled}
          className={cn(
            'relative flex h-16 w-16 items-center justify-center rounded-full border-2 transition-all',
            isRecording
              ? 'border-cyan-400/50 bg-cyan-500/15 animate-mic-pulse'
              : 'border-cyan-500/20 bg-cyan-500/10 hover:border-cyan-500/40 hover:bg-cyan-500/15 hover:glow-cyan-sm',
            disabled && 'opacity-40 cursor-not-allowed'
          )}
        >
          {isRecording ? (
            <Square
              className="h-6 w-6 text-cyan-400"
              fill="currentColor"
            />
          ) : (
            <Mic className="h-7 w-7 text-cyan-400" />
          )}
        </button>
      </div>

      <div className="text-center">
        <div className="font-mono text-2xl font-bold text-white tabular-nums">
          {formatTime(elapsed)}
        </div>

        <p className="mt-1 text-xs text-muted-foreground">
          {isRecording
            ? 'Recording in progress...'
            : 'Click to start recording'}
        </p>
      </div>

      <div className="w-full rounded-lg border border-cyan-500/15 bg-navy-900/50 p-3">
        <WaveformVisualizer
          bars={40}
          className="h-12"
          color="cyan"
          active={isRecording}
        />
      </div>

      <div className="flex items-center gap-3">
        {!isRecording ? (
          <button
            onClick={handleStart}
            disabled={disabled}
            className={cn(
              'inline-flex items-center gap-2 rounded-lg bg-cyan-500/15 border border-cyan-500/30 px-4 py-2 text-sm font-semibold text-cyan-400 transition-all hover:bg-cyan-500/25 hover:border-cyan-500/40',
              disabled && 'opacity-40 cursor-not-allowed'
            )}
          >
            <Mic className="h-4 w-4" />
            Start Recording
          </button>
        ) : (
          <button
            onClick={handleStop}
            className="inline-flex items-center gap-2 rounded-lg bg-red-500/15 border border-red-500/30 px-4 py-2 text-sm font-semibold text-red-400 transition-all hover:bg-red-500/25 hover:border-red-400"
          >
            <Square
              className="h-4 w-4"
              fill="currentColor"
            />
            Stop Recording
          </button>
        )}
      </div>
    </div>
  );
}