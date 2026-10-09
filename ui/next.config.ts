import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: '100gb',
    },
    middlewareClientMaxBodySize: '100gb',
  },
  serverExternalPackages: ['systeminformation'],
  webpack: config => {
    config.resolve ??= {};
    config.resolve.symlinks = false;
    return config;
  },
};

export default nextConfig;
